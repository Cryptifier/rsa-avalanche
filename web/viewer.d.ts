/* tslint:disable */
/* eslint-disable */

/**
 * Starts the egui viewer in a web canvas.
 *
 * # Parameters
 * - `canvas_id`: DOM id for the canvas element.
 *
 * # Returns
 * - `Result<(), JsValue>`: `Ok(())` on success.
 *
 * # Expected Output
 * - Attaches the viewer to the target canvas.
 */
export function start(canvas_id: string): Promise<void>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly start: (a: number, b: number) => any;
    readonly main: (a: number, b: number) => number;
    readonly wasm_bindgen__closure__destroy__h18e1474df18c514c: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h7471c2e7b1d10559: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h01cd4c9e5bdf191c: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__hc74aa43734449878: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h74f52eb314427e47: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h72f8c373ec23eeb4: (a: number, b: number) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h239b9fafb8c00785: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hde7f59a715426a44: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1402f82fa3ce511e: (a: number, b: number) => number;
    readonly wasm_bindgen__convert__closures_____invoke__hcb83bff2afdaf2cd: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
