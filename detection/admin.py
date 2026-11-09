from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "label",
        "formatted_score",
        "created_at",
    )

    list_filter = ("label", "created_at")
    search_fields = ("user__username", "label")
    ordering = ("-created_at", "-id")
    date_hierarchy = "created_at"
    list_select_related = ("user",)

    readonly_fields = (
        "user",
        "label",
        "score",
        "created_at",
    )

    @admin.display(description="Model score (%)", ordering="score")
    def formatted_score(self, obj):
        return f"{obj.score:.2f}%"

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False