mod rasm;

pub enum EncoderBackend {
    RasmBackend
}

pub fn encode(backend: EncoderBackend, input: &str) -> Vec<u8> {
    match backend {
        EncoderBackend::RasmBackend => rasm::encode(input)
    }
}
