from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils.translation import override

from .forms import LeafUploadForm


class LanguageTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="language_test",
            password="LeafCare-test-pass-2026",
        )
        self.client.force_login(self.user)

    def select_language(self, language):
        return self.client.post(
            reverse("set_language"),
            {
                "language": language,
                "next": reverse("detection:home"),
            },
        )

    def test_japanese_selection_sets_cookie_and_translates_home(self):
        response = self.select_language("ja")

        self.assertRedirects(
            response,
            reverse("detection:home"),
            fetch_redirect_response=False,
        )
        self.assertEqual(
            response.cookies[settings.LANGUAGE_COOKIE_NAME].value,
            "ja",
        )

        response = self.client.get(reverse("detection:home"))

        self.assertContains(response, 'lang="ja"')
        self.assertContains(response, "葉を解析する")
        self.assertContains(response, "葉の写真を選択")

    def test_switch_back_to_english(self):
        self.select_language("ja")
        self.select_language("en")

        response = self.client.get(reverse("detection:home"))

        self.assertContains(response, 'lang="en"')
        self.assertContains(response, "Analyze Leaf")
        self.assertContains(response, "Choose a leaf photo")
        self.assertNotContains(response, "葉を解析する")

    def test_japanese_dashboard(self):
        self.select_language("ja")

        response = self.client.get(
            reverse("detection:dashboard")
        )

        self.assertContains(response, "解析件数")
        self.assertContains(response, "最近の解析")

    def test_form_label_follows_active_language(self):
        with override("ja"):
            form = LeafUploadForm()
            self.assertEqual(
                form.fields["image"].label,
                "葉の写真を選択",
            )

        with override("en"):
            form = LeafUploadForm()
            self.assertEqual(
                form.fields["image"].label,
                "Choose a leaf photo",
            )