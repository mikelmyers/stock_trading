"""Alternative-data ingestion: orthogonal alpha axes the price-only model lacks.

Cross-sectional price-only ranking showed zero skill and free fundamentals were
noise (see PORTFOLIO_FINDINGS.md) — the remaining levers are data sources
orthogonal to price. Each module here normalizes one free source into a tidy,
POINT-IN-TIME-safe frame and exposes an ``attach_*`` that joins its features
onto the event dataset. training/augment_alt.py runs the verdict experiment.
"""
