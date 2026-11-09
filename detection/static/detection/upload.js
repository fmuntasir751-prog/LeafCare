const form = document.getElementById("upload-form");

if (form) {
    const input = form.querySelector('input[type="file"]');
    const preview = document.getElementById("selected-preview");
    const button = document.getElementById("analyze-button");
    const status = document.getElementById("upload-status");

    let previewUrl = null;

    function clearPreview() {
        preview.hidden = true;
        preview.removeAttribute("src");

        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
    }

    function validateFile() {
        const file = input.files[0];
        input.setCustomValidity("");

        if (!file) {
            return false;
        }

        if (!["image/jpeg", "image/png"].includes(file.type)) {
            input.setCustomValidity("Choose a JPEG or PNG image.");
            return false;
        }

        if (file.size > 5 * 1024 * 1024) {
            input.setCustomValidity("Image must be 5 MB or smaller.");
            return false;
        }

        return true;
    }

    input.addEventListener("change", () => {
        clearPreview();
        status.textContent = "";

        if (!validateFile()) {
            if (input.files.length) {
                status.textContent = input.validationMessage;
                input.reportValidity();
            }
            return;
        }

        previewUrl = URL.createObjectURL(input.files[0]);
        preview.src = previewUrl;
        preview.hidden = false;
    });

    form.addEventListener("submit", (event) => {
        if (!validateFile() || !form.checkValidity()) {
            event.preventDefault();
            form.reportValidity();
            return;
        }

        button.disabled = true;
        button.textContent = "Analyzing…";
        status.textContent = "Please wait while LeafCare analyzes your photo.";
        form.setAttribute("aria-busy", "true");
    });

    window.addEventListener("pageshow", () => {
        button.disabled = false;
        button.textContent = "Analyze Leaf";
        status.textContent = "";
        form.removeAttribute("aria-busy");
    });
}