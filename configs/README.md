# Configuration notes

## `model.jepa.simple_recurrence_steps`

- **Type:** integer (`>= 0`)
- **Default:** `0` (disabled)
- **Purpose:** When greater than zero, the tapped transformer block at `model.jepa.tap_layer` is reapplied that many times to iteratively refine the representation before the JEPA objective samples anchor/target pairs.
- **Training impact:** Each additional step costs one extra forward pass of the tapped block over `(batch, seq, dim)`. Gradients from the JEPA loss continue to update only the tapped block (respecting `grad_barrier`) plus the JEPA heads.
- **Evaluation:** The same refined tensor is used during validation so student and EMA teacher operate at the same depth.

Set the value inside your config YAML, e.g.

```yaml
model:
  jepa:
    simple_recurrence_steps: 2
    recur_at: -3  # optional override (defaults to tap_layer)
```

## `model.jepa.recur_at`

- **Type:** integer (layer index, supports negative indexing).
- **Default:** matches `tap_layer`.
- **Purpose:** Chooses which transformer block is rerun during the recurrence loop. By default the same layer that supplies the tap is reused, but you can point to another block if you want to experiment with a different "thinking" depth while still reading targets from the tap layer.

During training the student prediction uses the recurrent features, while the teacher target continues to read from the original (non-recurrent) tap activations, so gradients remain focused on the tapped block and JEPA heads.
