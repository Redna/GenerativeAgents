
// export TILESIZE constant with 32
export const UPDATE_INTERVAL_MS = 8_000;
export const TILESIZE = 32;
export const MOVEMENT_SPEED=2;

export function toTilePosition(x: number, y: number): { col: number, row: number } {
    return {
        col: Math.floor(x / TILESIZE),
        row: Math.floor(y / TILESIZE)
    }
}

export function toPixelPosition(col: number, row: number): { x: number, y: number } {
    return {
        x: col * TILESIZE + TILESIZE / 2,
        y: row * TILESIZE + TILESIZE
    }
}