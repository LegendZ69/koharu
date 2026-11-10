use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use koharu_scene::EntityId;

use crate::Stage;

/// How translation jobs group pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Plan {
    /// Up to this many pages per job.
    Pages(usize),
    /// Every page in one job.
    All,
    /// Pages up to this estimated source-token budget per job.
    Budget(usize),
}

/// The work a page has left after one of its stages completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Remaining {
    /// Detection, OCR, or inpainting.
    Local,
    /// Only translation; the page no longer occupies the window.
    Translation,
    Nothing,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum WorkState {
    Pending,
    Running,
    Finished,
}

struct StageWork {
    stage: Stage,
    state: WorkState,
}

struct PageWork {
    page: EntityId,
    entered: bool,
    stages: Vec<StageWork>,
}

impl PageWork {
    fn finished(&self) -> bool {
        self.stages
            .iter()
            .all(|work| work.state == WorkState::Finished)
    }

    fn remaining(&self) -> Remaining {
        if self
            .stages
            .iter()
            .any(|work| work.stage != Stage::Translation && work.state != WorkState::Finished)
        {
            Remaining::Local
        } else if self.finished() {
            Remaining::Nothing
        } else {
            Remaining::Translation
        }
    }

    fn pending(&self, stage: Stage) -> bool {
        self.stages
            .iter()
            .any(|work| work.stage == stage && work.state == WorkState::Pending)
    }

    fn ready(&self, stage: Stage) -> bool {
        let Some(prerequisite) = prerequisite(stage) else {
            return true;
        };
        self.stages
            .iter()
            .find(|work| work.stage == prerequisite)
            .is_none_or(|work| work.state == WorkState::Finished)
    }

    fn start(&mut self, stage: Stage) {
        self.entered = true;
        if let Some(work) = self.stages.iter_mut().find(|work| work.stage == stage) {
            work.state = WorkState::Running;
        }
    }
}

pub(crate) struct Scheduler {
    pages: Vec<PageWork>,
    page_index: BTreeMap<EntityId, usize>,
    page_window: usize,
    plan: Plan,
    head: usize,
    total: usize,
    blocked: bool,
}

impl Scheduler {
    pub(crate) fn new(pages: &[EntityId], stages: &[Stage], plan: Plan) -> Self {
        let pages = pages
            .iter()
            .map(|page| PageWork {
                page: *page,
                entered: false,
                stages: stages
                    .iter()
                    .map(|stage| StageWork {
                        stage: *stage,
                        state: WorkState::Pending,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let total = pages.len().saturating_mul(stages.len());
        Self {
            page_index: pages
                .iter()
                .enumerate()
                .map(|(index, page)| (page.page, index))
                .collect(),
            pages,
            page_window: stages.len().max(1),
            plan,
            head: 0,
            total,
            blocked: false,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Whether the last scan skipped a page because another run holds it.
    pub(crate) fn blocked(&self) -> bool {
        self.blocked
    }

    /// Starts one job of a stage that is not `busy`: one page for detection, OCR, or
    /// inpainting, or a batch of pages for translation. `enter` claims a page the first time it
    /// would start; a refused page is skipped for the rest of this scan. `held` tells, without
    /// claiming, whether another run holds a page this run has not entered. A translation batch
    /// leaves out such a page, as it does a refused one. `weight` sizes a page for
    /// `Plan::Budget` and is called only for pages whose translation prerequisite finished.
    pub(crate) fn start_next(
        &mut self,
        busy: &BTreeSet<Stage>,
        mut enter: impl FnMut(EntityId) -> bool,
        held: impl Fn(EntityId) -> bool,
        mut weight: impl FnMut(EntityId) -> usize,
    ) -> Option<(Stage, Vec<EntityId>)> {
        let translating = busy.contains(&Stage::Translation);
        let mut refused = BTreeSet::new();
        let batch = if translating {
            None
        } else {
            // A batch does not wait for a page another run holds.
            refused.extend((self.head..self.pages.len()).filter(|&index| {
                let page = &self.pages[index];
                !page.entered && page.pending(Stage::Translation) && held(page.page)
            }));
            self.translation_batch(&mut weight, &refused)
        };
        self.blocked = !refused.is_empty();
        // A ready batch is ordered by its oldest page, like a page-local job of that page.
        let oldest = batch.as_ref().map_or(self.pages.len(), |batch| batch[0]);
        let (before, after) = (self.head..oldest, oldest..self.pages.len());
        let known = refused.len();
        if let Some(job) = self.start_page_local(busy, &mut enter, &mut refused, before) {
            return Some(job);
        }
        // Another run can take a page between `held` and `enter`. A batch that waited for that
        // page is formed again without it, so this run's pages can finish and release their
        // leases.
        let batch = match batch {
            None if !translating && refused.len() > known => {
                self.translation_batch(&mut weight, &refused)
            }
            batch => batch,
        };
        if let Some(batch) = batch
            && let Some(job) = self.start_translation(batch, &mut enter, &mut weight, &mut refused)
        {
            return Some(job);
        }
        self.start_page_local(busy, &mut enter, &mut refused, after)
    }

    fn start_page_local(
        &mut self,
        busy: &BTreeSet<Stage>,
        enter: &mut impl FnMut(EntityId) -> bool,
        refused: &mut BTreeSet<usize>,
        range: Range<usize>,
    ) -> Option<(Stage, Vec<EntityId>)> {
        // A page waiting only for a translation batch is parked outside the window, so a
        // batch larger than the window can still fill.
        let active = self.pages[self.head..]
            .iter()
            .filter(|page| page.entered && page.remaining() == Remaining::Local)
            .count();
        for index in range {
            let page = &self.pages[index];
            if !page.entered && active >= self.page_window {
                continue;
            }
            let Some(stage) = page
                .stages
                .iter()
                .find(|work| {
                    work.stage != Stage::Translation
                        && work.state == WorkState::Pending
                        && !busy.contains(&work.stage)
                        && page.ready(work.stage)
                })
                .map(|work| work.stage)
            else {
                continue;
            };
            if !page.entered && !enter(page.page) {
                refused.insert(index);
                self.blocked = true;
                continue;
            }
            let page = &mut self.pages[index];
            page.start(stage);
            return Some((stage, vec![page.page]));
        }
        None
    }

    /// Enters the pages of `batch` and starts it; a refused page leaves the batch, which is
    /// formed again without it.
    fn start_translation(
        &mut self,
        mut batch: Vec<usize>,
        enter: &mut impl FnMut(EntityId) -> bool,
        weight: &mut impl FnMut(EntityId) -> usize,
        refused: &mut BTreeSet<usize>,
    ) -> Option<(Stage, Vec<EntityId>)> {
        loop {
            let mut entered = true;
            for &index in &batch {
                let page = &mut self.pages[index];
                if !page.entered {
                    if enter(page.page) {
                        page.entered = true;
                    } else {
                        refused.insert(index);
                        entered = false;
                    }
                }
            }
            if entered {
                let mut pages = Vec::with_capacity(batch.len());
                for index in batch {
                    let page = &mut self.pages[index];
                    page.start(Stage::Translation);
                    pages.push(page.page);
                }
                return Some((Stage::Translation, pages));
            }
            self.blocked = true;
            batch = self.translation_batch(weight, refused)?;
        }
    }

    /// The next translation batch, in project order, or `None` while it must wait.
    fn translation_batch(
        &self,
        weight: &mut impl FnMut(EntityId) -> usize,
        refused: &BTreeSet<usize>,
    ) -> Option<Vec<usize>> {
        let candidates = (self.head..self.pages.len()).filter(|index| {
            !refused.contains(index) && self.pages[*index].pending(Stage::Translation)
        });
        let ready = |index: usize| self.pages[index].ready(Stage::Translation);
        let batch = match self.plan {
            Plan::Pages(pages) => candidates.take(pages.max(1)).collect::<Vec<_>>(),
            Plan::All => candidates.collect(),
            Plan::Budget(budget) => {
                let mut batch = Vec::new();
                let mut total = 0_usize;
                for index in candidates {
                    if !ready(index) {
                        // Below its budget, a batch waits for the next page's prerequisite.
                        return (!batch.is_empty() && total >= budget).then_some(batch);
                    }
                    let page_weight = weight(self.pages[index].page);
                    if !batch.is_empty() && total.saturating_add(page_weight) > budget {
                        return Some(batch);
                    }
                    batch.push(index);
                    total = total.saturating_add(page_weight);
                }
                return (!batch.is_empty()).then_some(batch);
            }
        };
        (!batch.is_empty() && batch.iter().all(|&index| ready(index))).then_some(batch)
    }

    pub(crate) fn complete_stage(&mut self, page: EntityId, stage: Stage) -> Remaining {
        let Some(&page_index) = self.page_index.get(&page) else {
            return Remaining::Local;
        };
        let page = &mut self.pages[page_index];
        if let Some(work) = page.stages.iter_mut().find(|work| work.stage == stage) {
            work.state = WorkState::Finished;
        }
        let remaining = page.remaining();
        while self.head < self.pages.len() && self.pages[self.head].finished() {
            self.head += 1;
        }
        remaining
    }
}

const fn prerequisite(stage: Stage) -> Option<Stage> {
    match stage {
        Stage::Detection => None,
        Stage::Ocr | Stage::Inpainting => Some(Stage::Detection),
        Stage::Translation => Some(Stage::Ocr),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    fn pages(count: usize) -> Vec<EntityId> {
        (0..count).map(|_| EntityId::new()).collect()
    }

    fn start(scheduler: &mut Scheduler, busy: &BTreeSet<Stage>) -> Option<(Stage, Vec<EntityId>)> {
        scheduler.start_next(busy, |_| true, |_| false, |_| 0)
    }

    #[test]
    fn starts_pages_in_order_and_models_independently() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL, Plan::Pages(1));
        let mut busy = BTreeSet::new();

        let first = start(&mut scheduler, &busy).unwrap();
        assert_eq!(first, (Stage::Detection, vec![pages[0]]));
        busy.insert(Stage::Detection);
        assert!(start(&mut scheduler, &busy).is_none());

        busy.clear();
        assert_eq!(
            scheduler.complete_stage(pages[0], Stage::Detection),
            Remaining::Local
        );
        let ocr = start(&mut scheduler, &busy).unwrap();
        busy.insert(ocr.0);
        let inpainting = start(&mut scheduler, &busy).unwrap();
        busy.insert(inpainting.0);
        let next_page = start(&mut scheduler, &busy).unwrap();
        busy.insert(next_page.0);
        assert_eq!(ocr, (Stage::Ocr, vec![pages[0]]));
        assert_eq!(inpainting, (Stage::Inpainting, vec![pages[0]]));
        assert_eq!(next_page, (Stage::Detection, vec![pages[1]]));

        assert_eq!(
            scheduler.complete_stage(pages[0], Stage::Ocr),
            Remaining::Local
        );
        busy.remove(&Stage::Ocr);
        let translation = start(&mut scheduler, &busy).unwrap();
        assert_eq!(translation, (Stage::Translation, vec![pages[0]]));
        assert!(busy.contains(&Stage::Detection));
        assert!(busy.contains(&Stage::Inpainting));
    }

    #[test]
    fn sliding_window_backpressures_fast_upstream_models() {
        let pages = pages(4);
        let stages = [Stage::Detection, Stage::Ocr, Stage::Inpainting];
        let mut scheduler = Scheduler::new(&pages, &stages, Plan::Pages(1));
        let mut busy = BTreeSet::new();

        assert_eq!(
            start(&mut scheduler, &busy),
            Some((Stage::Detection, vec![pages[0]]))
        );
        assert_eq!(
            scheduler.complete_stage(pages[0], Stage::Detection),
            Remaining::Local
        );
        let ocr = start(&mut scheduler, &busy).unwrap();
        busy.insert(ocr.0);
        let inpainting = start(&mut scheduler, &busy).unwrap();
        busy.insert(inpainting.0);

        for page in &pages[1..3] {
            assert_eq!(
                start(&mut scheduler, &busy),
                Some((Stage::Detection, vec![*page]))
            );
            assert_eq!(
                scheduler.complete_stage(*page, Stage::Detection),
                Remaining::Local
            );
        }
        assert!(start(&mut scheduler, &busy).is_none());

        assert_eq!(
            scheduler.complete_stage(pages[0], Stage::Ocr),
            Remaining::Local
        );
        assert_eq!(
            scheduler.complete_stage(pages[0], Stage::Inpainting),
            Remaining::Nothing
        );
        busy.clear();
        assert_eq!(
            start(&mut scheduler, &busy),
            Some((Stage::Ocr, vec![pages[1]]))
        );
    }

    #[test]
    fn a_ready_batch_starts_before_work_on_newer_pages() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL, Plan::Pages(1));
        let mut busy = BTreeSet::new();

        assert_eq!(
            start(&mut scheduler, &busy),
            Some((Stage::Detection, vec![pages[0]]))
        );
        scheduler.complete_stage(pages[0], Stage::Detection);
        for stage in [Stage::Ocr, Stage::Inpainting, Stage::Detection] {
            let (started, _) = start(&mut scheduler, &busy).unwrap();
            assert_eq!(started, stage);
            busy.insert(started);
        }
        scheduler.complete_stage(pages[0], Stage::Ocr);
        scheduler.complete_stage(pages[1], Stage::Detection);
        busy.remove(&Stage::Ocr);
        busy.remove(&Stage::Detection);

        assert_eq!(
            start(&mut scheduler, &busy),
            Some((Stage::Translation, vec![pages[0]]))
        );
        busy.insert(Stage::Translation);
        assert_eq!(
            start(&mut scheduler, &busy),
            Some((Stage::Ocr, vec![pages[1]]))
        );
    }

    /// Runs every job to completion, oldest first, and returns the jobs in start order.
    fn drive(scheduler: &mut Scheduler) -> Vec<(Stage, Vec<EntityId>)> {
        let mut busy = BTreeSet::new();
        let mut running = Vec::new();
        let mut started = Vec::new();
        loop {
            while let Some(job) = start(scheduler, &busy) {
                busy.insert(job.0);
                running.push(job.clone());
                started.push(job);
            }
            if running.is_empty() {
                return started;
            }
            let (stage, pages) = running.remove(0);
            busy.remove(&stage);
            for page in pages {
                scheduler.complete_stage(page, stage);
            }
        }
    }

    #[test]
    fn pages_waiting_for_one_batch_do_not_fill_the_window() {
        let pages = pages(6);
        let mut scheduler = Scheduler::new(&pages, &Stage::ALL, Plan::All);

        let jobs = drive(&mut scheduler);

        let translations = jobs
            .iter()
            .filter(|(stage, _)| *stage == Stage::Translation)
            .collect::<Vec<_>>();
        assert_eq!(translations, [&(Stage::Translation, pages.clone())]);
        assert_eq!(
            jobs.iter().map(|(_, pages)| pages.len()).sum::<usize>(),
            scheduler.total()
        );
    }

    #[test]
    fn fixed_batches_wait_for_their_pages_and_end_with_the_tail() {
        let pages = pages(3);
        let mut scheduler =
            Scheduler::new(&pages, &[Stage::Ocr, Stage::Translation], Plan::Pages(2));
        let mut busy = BTreeSet::from([Stage::Ocr]);

        assert_eq!(
            start(&mut scheduler, &BTreeSet::new()),
            Some((Stage::Ocr, vec![pages[0]]))
        );
        assert!(start(&mut scheduler, &busy).is_none());
        assert_eq!(
            scheduler.complete_stage(pages[0], Stage::Ocr),
            Remaining::Translation
        );
        assert_eq!(
            start(&mut scheduler, &BTreeSet::new()),
            Some((Stage::Ocr, vec![pages[1]]))
        );
        assert!(start(&mut scheduler, &busy).is_none());
        scheduler.complete_stage(pages[1], Stage::Ocr);
        assert_eq!(
            start(&mut scheduler, &BTreeSet::new()),
            Some((Stage::Translation, pages[..2].to_vec()))
        );
        assert_eq!(
            start(&mut scheduler, &BTreeSet::from([Stage::Translation])),
            Some((Stage::Ocr, vec![pages[2]]))
        );

        busy.insert(Stage::Translation);
        scheduler.complete_stage(pages[2], Stage::Ocr);
        assert!(start(&mut scheduler, &busy).is_none());
        for page in &pages[..2] {
            assert_eq!(
                scheduler.complete_stage(*page, Stage::Translation),
                Remaining::Nothing
            );
        }
        assert_eq!(
            start(&mut scheduler, &BTreeSet::new()),
            Some((Stage::Translation, vec![pages[2]]))
        );
    }

    #[test]
    fn budget_batches_close_before_the_page_that_would_exceed_them() {
        let pages = pages(5);
        let weights = BTreeMap::from([
            (pages[0], 400),
            (pages[1], 400),
            (pages[2], 3_000),
            (pages[3], 100),
            (pages[4], 100),
        ]);
        let mut scheduler = Scheduler::new(&pages, &[Stage::Translation], Plan::Budget(1_000));
        let busy = BTreeSet::new();

        for expected in [&pages[..2], &pages[2..3], &pages[3..]] {
            let (stage, batch) = scheduler
                .start_next(&busy, |_| true, |_| false, |page| weights[&page])
                .unwrap();
            assert_eq!((stage, batch.as_slice()), (Stage::Translation, expected));
            for page in batch {
                scheduler.complete_stage(page, stage);
            }
        }
        assert!(start(&mut scheduler, &busy).is_none());
    }

    #[test]
    fn budget_batches_wait_for_the_next_page_until_they_are_full() {
        let pages = pages(2);
        let mut scheduler = Scheduler::new(
            &pages,
            &[Stage::Ocr, Stage::Translation],
            Plan::Budget(1_000),
        );
        let mut weighed = Vec::new();

        assert_eq!(
            start(&mut scheduler, &BTreeSet::new()),
            Some((Stage::Ocr, vec![pages[0]]))
        );
        scheduler.complete_stage(pages[0], Stage::Ocr);
        assert_eq!(
            start(&mut scheduler, &BTreeSet::new()),
            Some((Stage::Ocr, vec![pages[1]]))
        );
        let waiting = scheduler.start_next(
            &BTreeSet::from([Stage::Ocr]),
            |_| true,
            |_| false,
            |page| {
                weighed.push(page);
                100
            },
        );
        assert!(waiting.is_none());
        assert_eq!(weighed, [pages[0]]);

        scheduler.complete_stage(pages[1], Stage::Ocr);
        assert_eq!(
            scheduler.start_next(&BTreeSet::new(), |_| true, |_| false, |_| 100),
            Some((Stage::Translation, pages.clone()))
        );
    }

    #[test]
    fn a_refused_page_is_skipped_and_reported_blocked() {
        let pages = pages(3);
        let held = pages[1];
        let busy = BTreeSet::new();
        let mut scheduler = Scheduler::new(&pages, &[Stage::Detection], Plan::Pages(1));
        let refuse = |scheduler: &mut Scheduler| {
            scheduler.start_next(&busy, |page| page != held, |_| false, |_| 0)
        };

        assert_eq!(
            refuse(&mut scheduler),
            Some((Stage::Detection, vec![pages[0]]))
        );
        scheduler.complete_stage(pages[0], Stage::Detection);
        assert_eq!(
            refuse(&mut scheduler),
            Some((Stage::Detection, vec![pages[2]]))
        );
        assert!(scheduler.blocked());
        scheduler.complete_stage(pages[2], Stage::Detection);
        assert!(refuse(&mut scheduler).is_none());
        assert!(scheduler.blocked());

        assert_eq!(
            start(&mut scheduler, &busy),
            Some((Stage::Detection, vec![held]))
        );
        assert!(!scheduler.blocked());

        let mut scheduler = Scheduler::new(&pages, &[Stage::Translation], Plan::All);
        assert_eq!(
            refuse(&mut scheduler),
            Some((Stage::Translation, vec![pages[0], pages[2]]))
        );
        assert!(scheduler.blocked());

        // `held` did not report the page, as when another run takes it after the check.
        let mut scheduler = Scheduler::new(&pages, &[Stage::Ocr, Stage::Translation], Plan::All);
        for page in [pages[0], pages[2]] {
            assert_eq!(refuse(&mut scheduler), Some((Stage::Ocr, vec![page])));
            scheduler.complete_stage(page, Stage::Ocr);
        }
        assert_eq!(
            refuse(&mut scheduler),
            Some((Stage::Translation, vec![pages[0], pages[2]]))
        );
        assert!(scheduler.blocked());
    }

    /// Drives runs of one project that share its page leases, the way Execution does: in each
    /// round every run starts all of its ready jobs and completes its oldest running one, and a
    /// page's lease is released once the page has nothing left. The second run starts `delay`
    /// rounds later. Once a batch's pages finished OCR and its run's previous batch finished,
    /// the batch must start before a later page of its run enters, also when it skipped a page
    /// the other run holds; a `Budget` batch below its budget also waits for the next page, so
    /// it is not checked. Returns how many work items each run completed.
    fn drive_together(runs: &mut [Scheduler; 2], delay: usize) -> [usize; 2] {
        let leased = RefCell::new(BTreeSet::new());
        let mut busy = [BTreeSet::new(), BTreeSet::new()];
        let mut running = [Vec::new(), Vec::new()];
        let mut completed = [0; 2];
        // Per run, numbered by event: when each page index entered and finished OCR, and when
        // the last translation job finished.
        let mut event = 0;
        let mut entered = [BTreeMap::new(), BTreeMap::new()];
        let mut ocr = [BTreeMap::new(), BTreeMap::new()];
        let mut translated = [0; 2];
        for round in 0.. {
            let mut progressed = false;
            for (run, scheduler) in runs.iter_mut().enumerate() {
                if run == 1 && round < delay {
                    progressed = true;
                    continue;
                }
                while let Some((stage, pages)) = scheduler.start_next(
                    &busy[run],
                    |page| leased.borrow_mut().insert(page),
                    |page| leased.borrow().contains(&page),
                    |_| 400,
                ) {
                    event += 1;
                    let indexes = pages
                        .iter()
                        .map(|page| scheduler.page_index[page])
                        .collect::<Vec<_>>();
                    for &index in &indexes {
                        entered[run].entry(index).or_insert(event);
                    }
                    if stage == Stage::Translation && !matches!(scheduler.plan, Plan::Budget(_)) {
                        let ready = indexes
                            .iter()
                            .filter_map(|index| ocr[run].get(index))
                            .fold(translated[run], |ready, &at| ready.max(at));
                        let last = indexes[indexes.len() - 1];
                        assert!(
                            entered[run].range(last + 1..).all(|(_, &at)| at < ready),
                            "run {run} entered a page after its ready batch {indexes:?} \
                             ({:?}, delay {delay})",
                            scheduler.plan
                        );
                    }
                    busy[run].insert(stage);
                    running[run].push((stage, pages));
                }
                if running[run].is_empty() {
                    // Execution ends a run that has nothing running unless a lease blocks it.
                    assert!(
                        completed[run] == scheduler.total() || scheduler.blocked(),
                        "run {run} stopped after {} of {}",
                        completed[run],
                        scheduler.total()
                    );
                    continue;
                }
                let (stage, pages) = running[run].remove(0);
                busy[run].remove(&stage);
                event += 1;
                for page in pages {
                    completed[run] += 1;
                    if stage == Stage::Ocr {
                        ocr[run].insert(scheduler.page_index[&page], event);
                    }
                    if scheduler.complete_stage(page, stage) == Remaining::Nothing {
                        leased.borrow_mut().remove(&page);
                    }
                }
                if stage == Stage::Translation {
                    translated[run] = event;
                }
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        completed
    }

    #[test]
    fn overlapping_runs_of_one_project_both_finish() {
        let pages = pages(8);
        let through_translation = [Stage::Detection, Stage::Ocr, Stage::Translation];
        for plan in [Plan::Budget(1_000), Plan::All, Plan::Pages(2)] {
            for stages in [&Stage::ALL[..], &through_translation[..]] {
                for (first, second) in [(&pages[..], &pages[2..6]), (&pages[..], &pages[..])] {
                    for delay in 0..4 {
                        let mut runs = [
                            Scheduler::new(first, stages, plan),
                            Scheduler::new(second, stages, plan),
                        ];
                        let completed = drive_together(&mut runs, delay);
                        assert_eq!(
                            completed,
                            [runs[0].total(), runs[1].total()],
                            "{plan:?}, {} and {} pages, delay {delay}",
                            first.len(),
                            second.len()
                        );
                    }
                }
            }
        }
    }
}
