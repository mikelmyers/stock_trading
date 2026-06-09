"""Runner system — aggressive low-float momentum classifier-trader.

Separate from the validated swing model. Trades the small-cap / low-float /
high-RVOL momentum "runner" setup: find A+ conditions, push hard, cap the loss,
win the fat tail. The classifier learns from its OWN trades (online), bootstrapped
by the known condition recipe; the discipline layer caps every loss from day one;
paper-money first so the cold-start tuition is free.
"""
