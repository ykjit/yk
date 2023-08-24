/*
 * Find the address of main.
 *
 * This is done in C due to this Rust bug:
 * https://github.com/rust-lang/rust/issues/101906
 */

extern int main(int argc, char **argv);

void *find_main() { return main; }
