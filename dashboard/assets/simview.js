// Simulator view: enlarge-in-page and fullscreen (client-side only).
// While enlarged or fullscreen the <img> switches to the HD stream (?q=hd),
// and back to the normal stream afterwards, so HD bandwidth is only used when visible.
(function () {
  function streamFor(img, hd) {
    var want = hd ? "/sim.mjpg?q=hd" : "/sim.mjpg";
    if (img.getAttribute("src") !== want) img.setAttribute("src", want);
  }
  function sync(panel) {
    var img = panel.querySelector(".sim-img");
    if (!img) return;
    var wrap = panel.querySelector(".sim-wrap");
    var full = document.fullscreenElement === wrap;
    var big = panel.classList.contains("sim-expanded");
    streamFor(img, full || big);
    var btn = panel.querySelector(".sim-btn-expand");
    if (btn) btn.textContent = big ? "⤡ Shrink" : "⤢ Enlarge";
  }
  function toggleFullscreen(panel) {
    var wrap = panel.querySelector(".sim-wrap");
    if (document.fullscreenElement) document.exitFullscreen();
    else if (wrap && wrap.requestFullscreen) wrap.requestFullscreen();
  }
  document.addEventListener("click", function (e) {
    var panel = e.target.closest && e.target.closest(".sim-panel");
    if (!panel) return;
    if (e.target.closest(".sim-btn-expand")) {
      panel.classList.toggle("sim-expanded");
      sync(panel);
      window.dispatchEvent(new Event("resize")); // let Plotly re-fit the reflowed panels
    } else if (e.target.closest(".sim-btn-full")) {
      toggleFullscreen(panel);
    }
  });
  document.addEventListener("dblclick", function (e) {
    var panel = e.target.closest && e.target.closest(".sim-panel");
    if (panel && e.target.closest(".sim-wrap")) toggleFullscreen(panel);
  });
  document.addEventListener("fullscreenchange", function () {
    document.querySelectorAll(".sim-panel").forEach(sync);
  });
})();
