from django import forms
from django.utils.translation import get_language


class LeafUploadForm(forms.Form):
    image = forms.ImageField(
        label="Choose a leaf photo",
        widget=forms.ClearableFileInput(
            attrs={"accept": "image/jpeg,image/png"}
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_japanese = (
            get_language() or "en"
        ).startswith("ja")

        if self.is_japanese:
            self.fields["image"].label = "葉の写真を選択"

    def clean_image(self):
        image = self.cleaned_data["image"]

        if image.size > 5 * 1024 * 1024:
            message = (
                "画像サイズは5 MB以下にしてください。"
                if self.is_japanese
                else "Image must be 5 MB or smaller."
            )
            raise forms.ValidationError(message)

        if image.image.format not in {"JPEG", "PNG"}:
            message = (
                "JPEGまたはPNG形式の画像をアップロードしてください。"
                if self.is_japanese
                else "Please upload a JPEG or PNG image."
            )
            raise forms.ValidationError(message)

        return image