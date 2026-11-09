from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .models import Prediction


class PredictionAccessTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()

        cls.owner = User.objects.create_user(
            username="owner_test",
            password="TestPassword!482",
        )
        cls.other = User.objects.create_user(
            username="other_test",
            password="TestPassword!593",
        )

        cls.prediction = Prediction.objects.create(
            user=cls.owner,
            label="Tomato — Early blight",
            score=76.30,
        )

    def test_anonymous_user_must_login(self):
        urls = [
            reverse("detection:home"),
            reverse("detection:history"),
            reverse("detection:export_history"),
            reverse(
                "detection:result",
                args=[self.prediction.pk],
            ),
        ]

        for url in urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 302)
                self.assertTrue(
                    response.url.startswith(
                        reverse("detection:login") + "?next="
                    )
                )

    def test_other_user_cannot_open_result(self):
        self.client.force_login(self.other)

        response = self.client.get(
            reverse(
                "detection:result",
                args=[self.prediction.pk],
            )
        )

        self.assertEqual(response.status_code, 404)

    def test_history_contains_only_own_records(self):
        self.client.force_login(self.other)

        own_prediction = Prediction.objects.create(
            user=self.other,
            label="Tomato — Healthy",
            score=92.50,
        )

        response = self.client.get(
            reverse("detection:history")
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item.pk for item in response.context["predictions"]],
            [own_prediction.pk],
        )

    def test_csv_excludes_other_users_records(self):
        self.client.force_login(self.other)

        Prediction.objects.create(
            user=self.other,
            label="Tomato — Healthy",
            score=92.50,
        )

        response = self.client.get(
            reverse("detection:export_history")
        )
        content = response.content.decode("utf-8-sig")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Tomato — Healthy", content)
        self.assertNotIn("Tomato — Early blight", content)

    def test_result_refresh_does_not_create_records(self):
        self.client.force_login(self.owner)

        url = reverse(
            "detection:result",
            args=[self.prediction.pk],
        )
        before = Prediction.objects.count()

        for _ in range(3):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)

        self.assertEqual(
            Prediction.objects.count(),
            before,
        )