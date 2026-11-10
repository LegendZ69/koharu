import { describe, expect, it } from 'vitest'

import { containCamera, resizeFrame, rotateFrame, widthCamera, widthScroll } from '@/lib/geometry'
import type { Frame } from '@koharu/bridge/protocol'

const frame: Frame = { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 }

describe('canvas fit geometry', () => {
  it('contains the page in the viewport, centered', () => {
    expect(containCamera(800, 600, 1000, 1500)).toEqual({ zoom: 0.4, translation: [200, 0] })
    expect(containCamera(0, 0, 1000, 1500)).toEqual({ zoom: 1, translation: [0, 0] })
  })

  // Physical px at dpr 2: insets are left 128, right 32, top 32, bottom 32.
  it('fits the width beside the tool rail and starts a tall page at its top', () => {
    expect(widthCamera(1760, 1200, 800, 3000, 2)).toEqual({ zoom: 2, translation: [128, 32] })
    expect(widthCamera(1760, 1200, 800, 200, 2)).toEqual({ zoom: 2, translation: [128, 400] })
    expect(widthCamera(150, 1200, 800, 3000, 2)).toEqual({ zoom: 1, translation: [0, 0] })
  })

  it('keeps the page row under the top inset when the viewport is refitted', () => {
    const scrolled = { zoom: 2, translation: [128, -968] as [number, number] }
    const refitted = widthCamera(3360, 1200, 800, 3000, 2, scrolled)

    expect(refitted).toEqual({ zoom: 4, translation: [128, -1968] })
    // Page row 500 sat at the top inset before and still does.
    expect(500 * refitted.zoom + refitted.translation[1]).toBe(32)
  })

  it('scrolls a width fit only between the page edges', () => {
    expect(widthScroll(100, 2, 3000, 1200, 2)).toBe(32)
    expect(widthScroll(-100, 2, 3000, 1200, 2)).toBe(-100)
    expect(widthScroll(-9000, 2, 3000, 1200, 2)).toBe(1200 - 32 - 6000)
    expect(widthScroll(-9000, 2, 200, 1200, 2)).toBe(400)
  })
})

describe('selection control geometry', () => {
  it('resizes from a handle while keeping the opposite edge fixed', () => {
    expect(resizeFrame(frame, 'e', { x: 140, y: 45 }, 1)).toEqual({
      ...frame,
      width: 130,
    })
    expect(resizeFrame(frame, 'nw', { x: 0, y: 10 }, 1)).toEqual({
      ...frame,
      x: 0,
      y: 10,
      width: 110,
      height: 60,
    })
  })

  it('resizes in the frame local axes when it is rotated', () => {
    const result = resizeFrame({ ...frame, angle_degrees: 90 }, 'e', { x: 60, y: 125 }, 1)
    expect(result.x).toBeCloseTo(-5)
    expect(result.y).toBeCloseTo(35)
    expect(result.width).toBeCloseTo(130)
    expect(result.height).toBe(50)
  })

  it('rotates around the frame center', () => {
    expect(rotateFrame(frame, { x: 60, y: -5 }, { x: 110, y: 45 })).toEqual({
      ...frame,
      angle_degrees: 90,
    })
  })
})
