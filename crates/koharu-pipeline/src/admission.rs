//! Admission state shared by every run and every runner generation of one `Pipeline`. Runs from
//! several projects, and runs started before and after a configuration change, all pass
//! through it, so its limits hold for the whole process.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex, Weak},
};

use koharu_scene::{EntityId, ProjectId};

use crate::{
    Stage,
    accelerator::{AcceleratorGate, AcceleratorPermit},
    resources::ResourceMonitor,
    stages::Stages,
};

pub(crate) struct Admission {
    stages: [tokio::sync::Semaphore; Stage::ALL.len()],
    accelerator: AcceleratorGate,
    generations: Mutex<Vec<Weak<Stages>>>,
    pages: Mutex<HashSet<(ProjectId, EntityId)>>,
    released: tokio::sync::watch::Sender<u64>,
}

impl Admission {
    pub(crate) fn new(device: &koharu_ml::Device, resources: Arc<ResourceMonitor>) -> Self {
        Self {
            stages: std::array::from_fn(|_| tokio::sync::Semaphore::new(1)),
            accelerator: AcceleratorGate::new(device, resources),
            generations: Mutex::new(Vec::new()),
            pages: Mutex::new(HashSet::new()),
            released: tokio::sync::watch::channel(0).0,
        }
    }

    /// Waits until no other job of `stage` runs in any run.
    pub(crate) async fn stage(&self, stage: Stage) -> tokio::sync::SemaphorePermit<'_> {
        self.stages[stage as usize]
            .acquire()
            .await
            .expect("stage admission is never closed")
    }

    pub(crate) fn accelerator(&self) -> &AcceleratorGate {
        &self.accelerator
    }

    /// Makes a runner generation's models reachable by out-of-memory recovery while it lives.
    pub(crate) fn register(&self, stages: &Arc<Stages>) {
        let mut generations = self
            .generations
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        generations.retain(|generation| generation.strong_count() > 0);
        generations.push(Arc::downgrade(stages));
    }

    /// Takes the accelerator lane back after `requester` ran out of memory in `stage`, and
    /// unloads every other model of every live generation: a run that started before a
    /// configuration change still holds the models of its own generation.
    pub(crate) async fn recover(&self, stage: Stage, requester: &Stages) -> AcceleratorPermit {
        self.accelerator
            .recover(|| {
                let generations = self
                    .generations
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .iter()
                    .filter_map(Weak::upgrade)
                    .collect::<Vec<_>>();
                // Every generation unloads; stopping at the first one that freed a model would
                // leave the others resident.
                let mut unloaded = false;
                for generation in &generations {
                    unloaded |= generation.unload_except(stage, requester);
                }
                unloaded
            })
            .await
    }

    /// Claims `page` of `project` for one run. Another run of the same project is refused until
    /// the lease drops, because its patches would invalidate each other's observations.
    pub(crate) fn try_lease(
        self: &Arc<Self>,
        project: ProjectId,
        page: EntityId,
    ) -> Option<PageLease> {
        self.pages
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert((project, page))
            .then(|| PageLease {
                admission: self.clone(),
                key: (project, page),
            })
    }

    /// Whether a run of `project` holds `page`, without claiming it.
    pub(crate) fn leased(&self, project: ProjectId, page: EntityId) -> bool {
        self.pages
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains(&(project, page))
    }

    /// Changes whenever a page lease is released.
    pub(crate) fn releases(&self) -> tokio::sync::watch::Receiver<u64> {
        self.released.subscribe()
    }
}

pub(crate) struct PageLease {
    admission: Arc<Admission>,
    key: (ProjectId, EntityId),
}

impl Drop for PageLease {
    fn drop(&mut self) {
        self.admission
            .pages
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.key);
        self.admission
            .released
            .send_modify(|generation| *generation = generation.wrapping_add(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_page_is_leased_by_one_run_of_its_project_at_a_time() {
        let device = koharu_ml::Device::cpu();
        let admission = Arc::new(Admission::new(&device, ResourceMonitor::new(&device)));
        let first = koharu_scene::Session::memory().await.unwrap().project_id();
        let second = koharu_scene::Session::memory().await.unwrap().project_id();
        let page = EntityId::new();
        let released = admission.releases();

        let lease = admission.try_lease(first, page).unwrap();
        assert!(admission.try_lease(first, page).is_none());
        let other_project = admission.try_lease(second, page).unwrap();
        assert!(!released.has_changed().unwrap());

        drop(lease);
        assert!(released.has_changed().unwrap());
        assert!(admission.try_lease(first, page).is_some());
        drop(other_project);
    }
}
