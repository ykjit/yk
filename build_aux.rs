// Code shared by `ykrt/build.rs` and `xtask/src/main.rs`.

/// Determine what kind of tracing the user has requested at compile time, by looking at the output
/// of `rustc --print cfg`. Ideally, we'd have the user set a environment variable, and then set
/// RUSTFLAGS accordingly, but you can't set arbitrary RUSTFLAGS from build.rs.
/// This doesn't look for `-Ctracer` in RUSTFLAGS as different backends may use different methods
/// of setting the tracing kind.
fn find_tracing_kind(rustflags: &str) -> String {
    let cfgs =
        std::process::Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string()))
            .args(&["--print", "cfg"])
            .args(rustflags.split(" "))
            .output()
            .unwrap()
            .stdout;
    let cfgs = String::from_utf8(cfgs).unwrap();

    let re = regex::Regex::new(r#"tracermode="([a-z]*)"\n"#).unwrap();
    let mut cgs = re.captures_iter(&cfgs);
    let tracing_kind = if let Some(caps) = cgs.next() {
        caps.get(1).unwrap().as_str()
    } else {
        panic!("Please choose a tracer by setting `RUSTFLAGS=\"-C tracer=<kind>\"`.");
    };
    if cgs.next().is_some() {
        panic!("Tracer mode was specified more than once in $RUSTFLAGS");
    }
    tracing_kind.to_owned()
}

/// Given the RUSTFLAGS for the external workspace, make flags for the internal one.
fn make_internal_rustflags(rustflags: &str) -> String {
    // Remove `-C tracer=<kind>`, as this would stifle optimisations.
    let re = regex::Regex::new(r"-C\s*tracer=[a-z]*").unwrap();
    re.replace_all(rustflags, "").to_string()
}
