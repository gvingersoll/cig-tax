"""Calibrate and solve the cigarette-tax supply-and-demand model."""

import json
from pathlib import Path

from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar


BASE_DIR = Path(__file__).resolve().parent
ESTIMATES_PATH = BASE_DIR / "results" / "estimates.json"
MODEL_PATH = BASE_DIR / "results" / "model.json"
TAX = 2.0

PLACEHOLDER_ESTIMATES = {
    "q0": 20,
    "p0": 6,
    "elasticity": -0.5,
    "passthrough": 0.9,
}


def load_estimates():
    if not ESTIMATES_PATH.exists():
        return PLACEHOLDER_ESTIMATES.copy()

    with ESTIMATES_PATH.open(encoding="utf-8") as estimates_file:
        estimates = json.load(estimates_file)

    required_keys = ("q0", "p0", "elasticity", "passthrough")
    missing_keys = [key for key in required_keys if key not in estimates]
    if missing_keys:
        raise ValueError(
            f"{ESTIMATES_PATH} is missing required keys: {', '.join(missing_keys)}"
        )

    return {key: float(estimates[key]) for key in required_keys}


def validate_estimates(estimates):
    q0 = estimates["q0"]
    p0 = estimates["p0"]
    elasticity = estimates["elasticity"]
    passthrough = estimates["passthrough"]

    if q0 <= 0 or p0 <= 0:
        raise ValueError("q0 and p0 must both be positive.")
    if elasticity >= 0:
        raise ValueError("elasticity must be negative for downward-sloping demand.")
    if not 0 < passthrough <= 1:
        raise ValueError("passthrough must be greater than 0 and at most 1.")


def solve_quantity(tax, demand_intercept, supply_intercept, demand_slope, supply_slope, q0):
    def tax_wedge(quantity):
        buyer_price = demand_intercept - demand_slope * quantity
        seller_price = supply_intercept + supply_slope * quantity
        return buyer_price - seller_price - tax

    if tax <= 0:
        return q0
    if tax >= demand_intercept - supply_intercept:
        return 0.0

    return brentq(tax_wedge, 0.0, q0)


def main():
    estimates = load_estimates()
    validate_estimates(estimates)

    q0 = estimates["q0"]
    p0 = estimates["p0"]
    elasticity = estimates["elasticity"]
    passthrough = estimates["passthrough"]

    demand_slope = -p0 / (q0 * elasticity)
    supply_slope = demand_slope * (1.0 / passthrough - 1.0)
    demand_intercept = p0 + demand_slope * q0
    supply_intercept = p0 - supply_slope * q0

    def demand_price(quantity):
        return demand_intercept - demand_slope * quantity

    def supply_price(quantity):
        return supply_intercept + supply_slope * quantity

    q_after = solve_quantity(
        TAX,
        demand_intercept,
        supply_intercept,
        demand_slope,
        supply_slope,
        q0,
    )
    buyer_price_after = demand_price(q_after)
    seller_price_after = supply_price(q_after)

    consumer_surplus_before = quad(
        demand_price, 0.0, q0
    )[0] - p0 * q0
    consumer_surplus_after = quad(
        demand_price, 0.0, q_after
    )[0] - buyer_price_after * q_after
    producer_surplus_before = p0 * q0 - quad(
        supply_price, 0.0, q0
    )[0]
    producer_surplus_after = seller_price_after * q_after - quad(
        supply_price, 0.0, q_after
    )[0]

    delta_consumer_surplus = consumer_surplus_after - consumer_surplus_before
    delta_producer_surplus = producer_surplus_after - producer_surplus_before
    revenue = TAX * q_after
    deadweight_loss = quad(
        lambda quantity: demand_price(quantity) - supply_price(quantity),
        q_after,
        q0,
    )[0]
    surplus_check = (
        delta_consumer_surplus
        + delta_producer_surplus
        + revenue
        + deadweight_loss
    )
    if abs(surplus_check) > 1e-8:
        raise ArithmeticError(
            f"Surplus accounting does not balance (residual {surplus_check})."
        )

    surplus_before_tax = demand_intercept - supply_intercept
    optimum = minimize_scalar(
        lambda tax: -tax
        * solve_quantity(
            tax,
            demand_intercept,
            supply_intercept,
            demand_slope,
            supply_slope,
            q0,
        ),
        bounds=(0.0, surplus_before_tax),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not optimum.success:
        raise RuntimeError(f"Revenue maximization failed: {optimum.message}")

    revenue_maximizing_tax = float(optimum.x)
    theoretical_revenue_maximizing_tax = surplus_before_tax / 2.0
    results = {
        "q0": q0,
        "p0": p0,
        "elasticity": elasticity,
        "passthrough": passthrough,
        "demand_slope": demand_slope,
        "supply_slope": supply_slope,
        "quantity": q_after,
        "buyers_price": buyer_price_after,
        "sellers_price": seller_price_after,
        "delta_consumer_surplus": delta_consumer_surplus,
        "delta_producer_surplus": delta_producer_surplus,
        "revenue": revenue,
        "deadweight_loss": deadweight_loss,
        "surplus_check": surplus_check,
        "revenue_maximizing_tax": revenue_maximizing_tax,
        "theoretical_revenue_maximizing_tax": theoretical_revenue_maximizing_tax,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("w", encoding="utf-8") as model_file:
        json.dump(results, model_file, indent=2)
        model_file.write("\n")

    for key in (
        "quantity",
        "buyers_price",
        "sellers_price",
        "delta_consumer_surplus",
        "delta_producer_surplus",
        "revenue",
        "deadweight_loss",
        "surplus_check",
        "revenue_maximizing_tax",
        "theoretical_revenue_maximizing_tax",
    ):
        print(f"{key}: {results[key]:.6f}")


if __name__ == "__main__":
    main()
