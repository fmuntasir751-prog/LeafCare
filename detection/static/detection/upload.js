const form = document.getElementById("upload-form");

if (form) {
    const input = form.querySelector('input[type="file"]');
    const preview = document.getElementById("selected-preview");
    const button = document.getElementById("analyze-button");
    const status = document.getElementById("upload-status");

    const isJapanese = document.documentElement.lang
        .toLowerCase()
        .startsWith("ja");

    const messages = isJapanese
        ? {
            chooseImage: "葉の写真を選択してください。",
            format: "JPEGまたはPNG形式の画像を選択してください。",
            size: "画像サイズは5 MB以下にしてください。",
            analyzing: "解析中…",
            waiting: "写真を解析しています。しばらくお待ちください。",
        }
        : {
            chooseImage: "Please choose a leaf photo.",
            format: "Choose a JPEG or PNG image.",
            size: "Image must be 5 MB or smaller.",
            analyzing: "Analyzing…",
            waiting: "Please wait while LeafCare analyzes your photo.",
        };

    const originalButtonText = button.textContent.trim();
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
            input.setCustomValidity(messages.chooseImage);
            return false;
        }

        if (!["image/jpeg", "image/png"].includes(file.type)) {
            input.setCustomValidity(messages.format);
            return false;
        }

        if (file.size > 5 * 1024 * 1024) {
            input.setCustomValidity(messages.size);
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
        button.textContent = messages.analyzing;
        status.textContent = messages.waiting;
        form.setAttribute("aria-busy", "true");
    });

    window.addEventListener("pageshow", () => {
        button.disabled = false;
        button.textContent = originalButtonText;
        status.textContent = "";
        form.removeAttribute("aria-busy");

        // Clear any validation message restored by browser history.
        input.setCustomValidity("");
    });
}