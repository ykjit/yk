fn main() {
    cc::Build::new()
        .file("src/test_helpers.c")
        .compile("ykcompile_test_helpers");
}
