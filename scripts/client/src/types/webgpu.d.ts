// Pull the WebGPU global types (GPUDevice, GPUTexture, navigator.gpu, …) into
// the project. They are not in lib.dom yet; @webgpu/types ships the ambient
// declarations. Referenced here (rather than via tsconfig "types") so the
// auto-included @types set is left untouched.
/// <reference types="@webgpu/types" />
