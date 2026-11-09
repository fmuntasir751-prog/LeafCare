from django import forms


class LeafUploadForm(forms.Form):
    image = forms.ImageField(
        label="Choose a leaf photo",
        widget=forms.ClearableFileInput(
            attrs={"accept": "image/jpeg,image/png"}
        ),
    )

    def clean_image(self):
        image = self.cleaned_data["image"]

        if image.size > 5 * 1024 * 1024:
            raise forms.ValidationError("Image must be 5 MB or smaller.")

        if image.image.format not in {"JPEG", "PNG"}:
            raise forms.ValidationError("Please upload a JPEG or PNG image.")

        return image