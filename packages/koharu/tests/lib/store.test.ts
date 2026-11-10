import { describe, expect, it } from 'vitest'

import { receiveDownload, receiveJob, useKoharuStore } from '@/lib/store'
import type { Job } from '@koharu/bridge/protocol'

describe('editor state', () => {
  it('drops finished and stopped jobs and retains failed ones until they are dismissed', () => {
    const job: Job = {
      id: 'job',
      state: 'running',
      completed: 0,
      total: 4,
      page: null,
      stage: null,
      model: null,
      error: null,
    }
    receiveJob(job)
    expect(useKoharuStore.getState().jobs.job).toBe(job)
    receiveJob({ ...job, state: 'finished', completed: 4 })
    expect(useKoharuStore.getState().jobs).toEqual({})

    receiveJob(job)
    receiveJob({ ...job, state: 'stopped' })
    expect(useKoharuStore.getState().jobs).toEqual({})

    receiveJob({ ...job, state: 'failed', error: 'out of memory' })
    expect(useKoharuStore.getState().jobs.job).toMatchObject({ state: 'failed' })
    useKoharuStore.getState().dismissJob('job')
    expect(useKoharuStore.getState().jobs).toEqual({})
  })

  it('retains failed downloads until they are dismissed', () => {
    useKoharuStore.setState({ downloads: {} })
    receiveDownload({
      state: 'failed',
      id: 9,
      name: 'weights.bin',
      completed: 0,
      total: 0,
      error: 'network unavailable',
    })
    expect(useKoharuStore.getState().downloads[9]).toMatchObject({
      state: 'failed',
      name: 'weights.bin',
    })

    useKoharuStore.getState().dismissDownload(9)
    expect(useKoharuStore.getState().downloads[9]).toBeUndefined()
  })

  it('defaults paint brush color to pure white (#FFFFFF)', () => {
    expect(useKoharuStore.getState().brush.color).toBe('#FFFFFF')
  })
})
