"""
Household Energy Coach Router.

Endpoints:
- GET /api/v1/coach/overview: Personal overview, historical trends, data-grounded insights, and recommendations.
- POST /api/v1/coach/simulate-appliance: What-if appliance simulation comparing baseline vs modified scenarios.
- GET /api/v1/coach/presets: Standard BEE/appliance presets for Indian households.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status as http_status

from backend.app.core.dependencies import get_current_user
from backend.app.schemas.coach import (
    ApplianceScenarioRequest,
    ApplianceScenarioResponse,
    DEFAULT_APPLIANCE_PRESETS,
    EnergyCoachOverviewResponse,
    HouseholdScenarioComparisonRequest,
    HouseholdScenarioComparisonResponse,
)
from backend.app.services.coach_service import coach_service

router = APIRouter(prefix="/coach", tags=["Household Energy Coach"])


@router.get(
    "/overview",
    response_model=EnergyCoachOverviewResponse,
    summary="Get Personalized Household Energy Overview & Insights",
    description=(
        "Returns personalized energy metrics, latest confirmed consumption, month-over-month trajectory, "
        "deterministic evidence-based insights, and actionable recommendations. "
        "Strictly restricted to the authenticated household's own confirmed records."
    ),
)
def get_energy_coach_overview(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> EnergyCoachOverviewResponse:
    """Retrieve personalized household energy coaching profile."""
    user_id = current_user["id"]
    try:
        overview = coach_service.get_household_overview(user_id=user_id)
        return overview
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while compiling your energy coaching profile: {str(exc)}",
        )


@router.post(
    "/simulate-appliance",
    response_model=ApplianceScenarioResponse,
    summary="Simulate Appliance Energy & Cost Impact",
    description=(
        "Simulates incremental appliance energy consumption using standard physics formulas: "
        "kWh = (watts × hours/day × days/month × quantity) / 1000. "
        "Computes differences between baseline and modified scenarios. "
        "Provides verified tariff cost calculations only when an official, verified tariff applies."
    ),
)
def simulate_appliance_scenario(
    payload: ApplianceScenarioRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ApplianceScenarioResponse:
    """Simulate appliance what-if scenario with verified tariff integration."""
    try:
        result = coach_service.simulate_appliance_scenario(request=payload)
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while running appliance simulation: {str(exc)}",
        )


@router.post(
    "/compare-household-scenario",
    response_model=HouseholdScenarioComparisonResponse,
    summary="Compare Household Forecast Baseline with Appliance What-If Scenario",
    description=(
        "Evaluates a multi-month household baseline forecast against a hypothetical appliance scenario. "
        "Calculates month-by-month energy reductions and cumulative savings using verified utility tariffs. "
        "Suppresses monetary amounts if provider tariff is unverified or expired."
    ),
)
def compare_household_scenario(
    payload: HouseholdScenarioComparisonRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> HouseholdScenarioComparisonResponse:
    """Compare household baseline forecast against appliance scenario."""
    try:
        result = coach_service.compare_household_scenario(request=payload)
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while evaluating scenario comparison: {str(exc)}",
        )



@router.get(
    "/presets",
    summary="Get Appliance Presets and Duty Cycle Benchmarks",
    description="Returns standard appliance presets, typical wattages, and duty cycle notes based on BEE benchmarks.",
)
def get_appliance_presets() -> Dict[str, Any]:
    """Return default presets for household appliances."""
    return DEFAULT_APPLIANCE_PRESETS
