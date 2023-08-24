fn main() {
    let mut comp = cc::Build::new();
    comp.file("src/find_main.c");
    comp.compile("find_main");
}
