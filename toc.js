// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="intro.html"><strong aria-hidden="true">1.</strong> Introduction</a></li><li class="chapter-item expanded "><a href="user/index.html"><strong aria-hidden="true">2.</strong> Using Yk</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="user/install.html"><strong aria-hidden="true">2.1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="user/interps.html"><strong aria-hidden="true">2.2.</strong> Available Interpreters</a></li></ol></li><li class="chapter-item expanded "><a href="dev/index.html"><strong aria-hidden="true">3.</strong> Development</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="dev/build_config.html"><strong aria-hidden="true">3.1.</strong> Configuring the build</a></li><li class="chapter-item expanded "><a href="dev/runtime_config.html"><strong aria-hidden="true">3.2.</strong> Run-time configuration</a></li><li class="chapter-item expanded "><a href="dev/debugging.html"><strong aria-hidden="true">3.3.</strong> Debugging</a></li><li class="chapter-item expanded "><a href="dev/profiling.html"><strong aria-hidden="true">3.4.</strong> Profiling</a></li><li class="chapter-item expanded "><a href="dev/understanding_traces.html"><strong aria-hidden="true">3.5.</strong> Understanding Traces</a></li><li class="chapter-item expanded "><a href="dev/gotchas.html"><strong aria-hidden="true">3.6.</strong> Gotchas</a></li></ol></li><li class="chapter-item expanded "><a href="internals/index.html"><strong aria-hidden="true">4.</strong> Internals</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="internals/debug_testing.html"><strong aria-hidden="true">4.1.</strong> Debugging / testing</a></li><li class="chapter-item expanded "><a href="internals/working_on_yk.html"><strong aria-hidden="true">4.2.</strong> Working on yk</a></li></ol></li><li class="chapter-item expanded "><div><strong aria-hidden="true">5.</strong> Contributing</div></li><li><ol class="section"><li class="chapter-item expanded "><a href="contributing/prs.html"><strong aria-hidden="true">5.1.</strong> Pull Requests</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
