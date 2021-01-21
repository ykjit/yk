use fm::FMBuilder;
use regex::Regex;
use ykshim_client::TirTrace;

/// Fuzzy matches the textual TIR for the trace `tt` with the pattern `ptn`.
pub fn assert_tir(ptn: &str, tt: &TirTrace) {
    let ptn_re = Regex::new(r"%.+?\b").unwrap(); // Names are words prefixed with `%`.
    let text_re = Regex::new(r"\$?.+?\b").unwrap(); // Any word optionally prefixed with `$`.
    let matcher = FMBuilder::new(ptn)
        .unwrap()
        .name_matcher(ptn_re, text_re)
        .distinct_name_matching(true)
        .build()
        .unwrap();

    let res = matcher.matches(&format!("{}", tt));
    if let Err(e) = res {
        panic!("{}", e);
    }
}
