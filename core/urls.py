from django.urls import path, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from rest_framework.routers import DefaultRouter
from .views import UploadViewSet, PlanViewSet

router = DefaultRouter()
router.register(r"uploads", UploadViewSet, basename="upload")
router.register(r"plans", PlanViewSet, basename="plan")

urlpatterns = [
    path("", include(router.urls)),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
