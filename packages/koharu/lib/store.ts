'use client'

import { create } from 'zustand'

import type {
  CanvasState,
  Download,
  EntityId,
  Job,
  JobId,
  Model,
  Frame,
  Preferences,
  Stage,
  StartupState,
  ModelResources,
  PageSize,
} from '@koharu/bridge/protocol'
import { toast } from '@koharu/ui/components/toast'

import type { CanvasFit } from './geometry'

export type CanvasTool = 'select' | 'text' | 'draw' | 'eraser' | 'color_picker' | 'remove' | 'pan'
export const MIN_BRUSH_DIAMETER = 1
export function maxBrushDiameter(size: PageSize): number {
  // Assume that for all sizes below 2048 width, 128 is sufficient.
  // For anything above, scale by percentage greater than 2048 width.
  return Math.round(Math.max(size.width / 2048, 1.0) * 128)
}

export function isBrushTool(tool: CanvasTool): boolean {
  return tool === 'draw' || tool === 'eraser' || tool === 'remove'
}

export interface CanvasBrush {
  diameter: number
  color: string
}
export type InspectorSection = 'copy' | 'type' | 'layers'
export type ShortcutAction = CanvasTool | 'fit' | 'fitWidth'
export type Shortcuts = Record<ShortcutAction, string>
export type PipelineScope = 'page' | 'selected-pages' | 'project'
export const pipelineStages: readonly Stage[] = ['detection', 'ocr', 'translation', 'inpainting']

interface KoharuStore {
  initialized: boolean
  preferences: Preferences | null
  translationModels: Model[]
  resources: ModelResources | null
  jobs: Record<string, Job>
  endedJobs: JobId[]
  downloads: Record<string, Download>
  camera: { zoom: number; translation: [number, number]; fit: CanvasFit | null }
  canvasPage: EntityId | null
  canvasRevision: number | null
  canvasGeneration: number
  canvasSize: [number, number]
  canvasViewport: [number, number]
  fitCanvasRequest: number
  fitMode: CanvasFit
  layerFrames: Record<EntityId, Frame>
  selectedLayers: EntityId[]
  selectedPages: EntityId[]
  tool: CanvasTool
  brush: CanvasBrush
  inspector: InspectorSection
  processingScope: PipelineScope
  processingStages: Stage[]
  settingsOpen: boolean
  shortcuts: Shortcuts
  selectPages: (pages: EntityId[]) => void
  showInspector: (section: InspectorSection) => void
  setProcessingScope: (scope: PipelineScope) => void
  setProcessingStages: (stages: Stage[]) => void
  setSettingsOpen: (open: boolean) => void
  selectLayers: (layers: EntityId[]) => void
  setTool: (tool: CanvasTool) => void
  setBrush: (brush: CanvasBrush) => void
  setShortcut: (action: ShortcutAction, key: string) => void
  requestCanvasFit: (fit: CanvasFit) => void
  dismissJob: (id: string) => void
  dismissDownload: (id: number) => void
}

export const defaultShortcuts: Shortcuts = {
  select: 'v',
  text: 't',
  draw: 'b',
  eraser: 'e',
  color_picker: 'i',
  remove: 'j',
  pan: 'h',
  fit: '0',
  fitWidth: 'w',
}

export const useKoharuStore = create<KoharuStore>()((set) => ({
  initialized: false,
  preferences: null,
  translationModels: [],
  resources: null,
  jobs: {},
  endedJobs: [],
  downloads: {},
  camera: { zoom: 1, translation: [0, 0], fit: 'page' },
  canvasPage: null,
  canvasRevision: null,
  canvasGeneration: 0,
  canvasSize: [0, 0],
  canvasViewport: [0, 0],
  fitCanvasRequest: 0,
  fitMode: 'page',
  layerFrames: {},
  selectedLayers: [],
  selectedPages: [],
  tool: 'select',
  brush: { diameter: 48, color: '#FFFFFF' },
  inspector: 'copy',
  processingScope: 'selected-pages',
  processingStages: [...pipelineStages],
  settingsOpen: false,
  shortcuts: defaultShortcuts,
  selectPages: (selectedPages) => set({ selectedPages: [...new Set(selectedPages)] }),
  showInspector: (inspector) => set({ inspector }),
  setProcessingScope: (processingScope) => set({ processingScope }),
  setProcessingStages: (processingStages) => set({ processingStages }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  selectLayers: (selectedLayers) => set({ selectedLayers: [...new Set(selectedLayers)] }),
  setTool: (tool) => set({ tool }),
  setBrush: (brush) => set({ brush }),
  setShortcut: (action, key) =>
    set((state) => ({
      shortcuts: { ...state.shortcuts, [action]: key.toLowerCase().slice(0, 1) },
    })),
  requestCanvasFit: (fitMode) =>
    set((state) => ({ fitMode, fitCanvasRequest: state.fitCanvasRequest + 1 })),
  dismissJob: (id) =>
    set((state) => {
      const jobs = { ...state.jobs }
      delete jobs[id]
      return { jobs }
    }),
  dismissDownload: (id) =>
    set((state) => {
      const downloads = { ...state.downloads }
      delete downloads[String(id)]
      return { downloads }
    }),
}))

export function receiveStartupState(state: StartupState): void {
  useKoharuStore.setState({
    initialized: true,
    preferences: state.preferences,
    jobs: byId(state.jobs),
    ...canvasSnapshot(state.canvas),
  })
}

export function receiveCanvas(canvas: CanvasState): void {
  useKoharuStore.setState(canvasSnapshot(canvas))
}

// A command's result and its job's messages arrive in no fixed order, so the ids of recently
// ended jobs tell a job that ended from one that was not reported yet.
const ENDED_JOB_LIMIT = 32

export function receiveJob(job: Job): void {
  useKoharuStore.setState((state) => {
    const jobs = { ...state.jobs }
    if (job.state === 'finished' || job.state === 'stopped') delete jobs[job.id]
    else jobs[job.id] = job
    if (job.state === 'running') return { jobs }
    return { jobs, endedJobs: [...state.endedJobs, job.id].slice(-ENDED_JOB_LIMIT) }
  })
}

export function receiveDownload(download: Download): void {
  useKoharuStore.setState((state) => {
    const downloads = { ...state.downloads }
    if (download.state === 'finished') delete downloads[String(download.id)]
    else downloads[String(download.id)] = download
    return { downloads }
  })
}

export function receivePreferences(preferences: Preferences): void {
  useKoharuStore.setState({ preferences })
}

export function receiveTranslationModels(translationModels: Model[]): void {
  useKoharuStore.setState({ translationModels })
}

export function receiveResources(resources: ModelResources): void {
  useKoharuStore.setState({ resources })
}

export function receiveError(message: string): void {
  toast.add({ type: 'error', title: 'Could not complete that action', description: message })
}

function canvasSnapshot(canvas: CanvasState) {
  return {
    canvasPage: canvas.page,
    canvasRevision: canvas.revision,
    canvasGeneration: canvas.generation,
    canvasSize: canvas.size,
    layerFrames: Object.fromEntries(
      canvas.element_frames.map(({ element, frame }) => [element, frame]),
    ),
  }
}

function byId<T extends { id: string | number }>(items: T[]): Record<string, T> {
  return Object.fromEntries(items.map((item) => [String(item.id), item]))
}
