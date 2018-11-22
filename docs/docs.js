$(function() { $(".rst-current-version").on("click", function() {
        fixGithubLink($('.injected a:contains(View)'))
        fixGithubLink($('.injected a:contains(Edit)')) }) })

function fixGithubLink(elem) { if (elem.length) elem.attr("href",
        elem.attr("href").replace(/\/home\/docs\/checkouts\/readthedocs.org\/.*?\/latest/,
        "")) }
