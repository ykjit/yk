use std::process::Command;

pub fn encode(input: &str) -> Vec<u8> {
    let output = Command::new("rasm2")
        .arg("-B")
        .arg(input)
        .output()
        .expect("failed to execute process");
    output.stdout
}
