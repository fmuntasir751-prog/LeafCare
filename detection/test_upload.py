from io import BytesIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse
from PIL import Image

from .models import Prediction
from .predictor import NotTomatoLeafError


def make_image():
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color="green").save(
        buffer, format="JPEG"
    )

    return SimpleUploadedFile(
        "test_leaf.jpg",
        buffer.getvalue(),
        content_type="image/jpeg",
    )


class UploadFlowTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="upload_test",
            password="TestPassword!482",
        )
        self.client.force_login(self.user)
        self.url = reverse("detection:home")

    @patch("detection.views.predict_leaf")
    def test_success_saves_and_redirects(self, mock_predict):
        mock_predict.return_value = {
            "label": "Tomato — Healthy",
            "score": 92.50,
        }

        response = self.client.post(
            self.url,
            {"image": make_image()},
        )

        prediction = Prediction.objects.get(user=self.user)

        self.assertEqual(prediction.label, "Tomato — Healthy")
        self.assertAlmostEqual(prediction.score, 92.50)
        mock_predict.assert_called_once()

        self.assertRedirects(
            response,
            reverse(
                "detection:result",
                args=[prediction.pk],
            ),
        )

        # Refreshing the result must not save another prediction.
        self.client.get(response.url)
        self.assertEqual(Prediction.objects.count(), 1)

    @patch("detection.views.predict_leaf")
    def test_invalid_file_is_rejected(self, mock_predict):
        invalid_file = SimpleUploadedFile(
            "fake.jpg",
            b"This is not an image.",
            content_type="image/jpeg",
        )

        response = self.client.post(
            self.url,
            {"image": invalid_file},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("image", response.context["form"].errors)
        mock_predict.assert_not_called()
        self.assertEqual(Prediction.objects.count(), 0)

    @patch(
        "detection.views.predict_leaf",
        side_effect=FileNotFoundError("Model missing"),
    )
    def test_missing_model_does_not_save(self, mock_predict):
        response = self.client.post(
            self.url,
            {"image": make_image()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "A required model file is unavailable.",
        )
        mock_predict.assert_called_once()
        self.assertEqual(Prediction.objects.count(), 0)


    @patch(
        "detection.views.predict_leaf",
        side_effect=NotTomatoLeafError(
            "This image was not accepted as a tomato leaf."
        ),
    )
    def test_rejected_image_does_not_save(self, mock_predict):
        response = self.client.post(
            self.url,
            {"image": make_image()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "image",
            response.context["form"].errors,
        )
        self.assertContains(
            response,
            "This image was not accepted as a tomato leaf.",
        )
        self.assertEqual(Prediction.objects.count(), 0)
        mock_predict.assert_called_once()