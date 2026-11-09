import csv
import json

from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.core.paginator import Paginator
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .forms import LeafUploadForm
from .models import Prediction
from .predictor import NotTomatoLeafError, predict_leaf


@require_http_methods(["GET", "POST"])
def register(request):
    if request.user.is_authenticated:
        return redirect("detection:home")

    form = UserCreationForm(
        request.POST if request.method == "POST" else None
    )

    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("detection:home")

    return render(
        request,
        "detection/register.html",
        {"form": form},
    )


@login_required
@require_http_methods(["GET", "POST"])
def home(request):
    form = LeafUploadForm()

    if request.method == "POST":
        form = LeafUploadForm(request.POST, request.FILES)

        if form.is_valid():
            image = form.cleaned_data["image"]

            try:
                result = predict_leaf(image)

            except NotTomatoLeafError as error:
                form.add_error("image", str(error))

            except FileNotFoundError:
                form.add_error(
                    None,
                    "A required model file is unavailable. "
                    "Please contact the administrator.",
                )

            except (OSError, RuntimeError, ValueError, KeyError):
                form.add_error(
                    None,
                    "Analysis could not be completed. "
                    "Please try another image or contact "
                    "the administrator.",
                )

            else:
                prediction = Prediction.objects.create(
                    user=request.user,
                    label=result["label"],
                    score=result["score"],
                )

                return redirect(
                    "detection:result",
                    pk=prediction.pk,
                )

    return render(
        request,
        "detection/home.html",
        {"form": form},
    )


@login_required
@require_http_methods(["GET"])
def prediction_result(request, pk):
    prediction = get_object_or_404(
        Prediction,
        pk=pk,
        user=request.user,
    )

    return render(
        request,
        "detection/result.html",
        {"prediction": prediction},
    )


@login_required
@require_http_methods(["GET"])
def history(request):
    choices = [
        ("healthy", "Tomato — Healthy"),
        ("early_blight", "Tomato — Early blight"),
        ("late_blight", "Tomato — Late blight"),
        ("leaf_mold", "Tomato — Leaf mold"),
    ]

    selected = request.GET.get("condition", "")
    labels = dict(choices)

    if selected not in labels:
        selected = ""

    predictions = Prediction.objects.filter(
        user=request.user
    ).order_by("-created_at", "-id")

    if selected:
        predictions = predictions.filter(
            label=labels[selected]
        )

    paginator = Paginator(predictions, 20)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(
        request,
        "detection/history.html",
        {
            "predictions": page_obj.object_list,
            "page_obj": page_obj,
            "choices": choices,
            "selected": selected,
            "total": paginator.count,
        },
    )


@login_required
@require_http_methods(["GET"])
def export_history(request):
    response = HttpResponse(
        content_type="text/csv; charset=utf-8"
    )

    response["Content-Disposition"] = (
        'attachment; filename="leafcare_history.csv"'
    )

    # UTF-8 BOM helps Excel display the text correctly.
    response.write("\ufeff")

    writer = csv.writer(response)
    writer.writerow([
        "Date / Time (Asia/Tokyo)",
        "Prediction",
        "Model Score (%)",
    ])

    predictions = Prediction.objects.filter(
        user=request.user
    ).order_by("-created_at", "-id")

    for prediction in predictions.iterator():
        writer.writerow([
            timezone.localtime(
                prediction.created_at
            ).strftime("%Y-%m-%d %H:%M:%S"),
            prediction.label,
            f"{prediction.score:.2f}",
        ])

    return response


@login_required
@require_http_methods(["GET"])
def dashboard(request):
    predictions = Prediction.objects.filter(
        user=request.user
    )

    total = predictions.count()
    healthy = predictions.filter(
        label="Tomato — Healthy"
    ).count()

    return render(
        request,
        "detection/dashboard.html",
        {
            "total": total,
            "healthy": healthy,
            "disease": total - healthy,
            "recent": predictions.order_by(
                "-created_at", "-id"
            )[:5],
        },
    )


@login_required
@require_http_methods(["GET"])
def performance(request):
    report = None
    rows = []

    try:
        path = (
            settings.BASE_DIR
            / "artifacts"
            / "test_metrics.json"
        )
        report = json.loads(
            path.read_text(encoding="utf-8")
        )

        for name, values in report["classification_report"].items():
            if name in {
                "accuracy",
                "macro avg",
                "weighted avg",
            }:
                continue

            rows.append({
                "name": name.replace("_", " "),
                "precision": values["precision"] * 100,
                "recall": values["recall"] * 100,
                "f1": values["f1-score"] * 100,
                "support": values["support"],
            })

        report["accuracy_percent"] = (
            report["accuracy"] * 100
        )

    except (OSError, ValueError, KeyError, TypeError):
        report = None
        rows = []

    return render(
        request,
        "detection/performance.html",
        {
            "report": report,
            "rows": rows,
        },
    )


@login_required
@require_http_methods(["GET"])
def confusion_matrix_image(request):
    path = (
        settings.BASE_DIR
        / "artifacts"
        / "confusion_matrix.png"
    )

    try:
        image = path.open("rb")
    except FileNotFoundError:
        raise Http404(
            "Confusion matrix is unavailable."
        ) from None

    return FileResponse(
        image,
        content_type="image/png",
    )