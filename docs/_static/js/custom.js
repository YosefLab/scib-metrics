// Article header buttons (see docs/_templates/header-buttons.html).

// Toggle full screen mode
function toggleFullScreen() {
    if (document.fullscreenElement || document.webkitFullscreenElement) {
        (document.exitFullscreen || document.webkitExitFullscreen).call(
            document,
        );
    } else {
        let root = document.documentElement;
        (root.requestFullscreen || root.webkitRequestFullscreen).call(root);
    }
}

document.addEventListener("click", function (event) {
    if (event.target.closest(".scib-fullscreen-button")) {
        toggleFullScreen();
    } else if (event.target.closest(".scib-print-pdf")) {
        // Browsers offer to save the printed page as PDF
        window.print();
    }
});

// Swap the icon to match the current state, also when leaving with Esc
function updateFullScreenIcon() {
    let isFull = Boolean(
        document.fullscreenElement || document.webkitFullscreenElement,
    );
    document
        .querySelectorAll(".scib-fullscreen-button i")
        .forEach(function (icon) {
            icon.classList.toggle("fa-expand", !isFull);
            icon.classList.toggle("fa-compress", isFull);
        });
}

document.addEventListener("fullscreenchange", updateFullScreenIcon);
document.addEventListener("webkitfullscreenchange", updateFullScreenIcon);
