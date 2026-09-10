# Figure palette

Tokens for the SVG figures in this directory, taken from the theme of
<https://radarsimx.com/radarsimx/radarsimpy/>. Use these when adding a new
figure so the documentation reads as one set.

## Colors

| Role | Token | Used for |
| --- | --- | --- |
| Blue (primary) | `#1e73be` | Rays, signal paths, the radar itself, the main subject of a figure |
| Green | `#1a7f37` | Environment surfaces, ground, the receive side, secondary annotations |
| Coral | `#cf4640` | Targets of interest, results, things being called out |
| Ink | `#0d1117` | Headings and emphasized labels |
| Muted text | `#6b7280` | Captions, axis names, secondary labels |
| Axis / wire | `#848d97` | Axis lines, connectors, arrow markers |
| Grid / leader | `#d0d7de` | Grid lines, leader lines, card borders |
| Panel border | `#d8dee4` | Inner panel outlines |
| Panel fill | `#f6f8fa` | Inner panel backgrounds |
| Card fill | `#ffffff` | The outer card background |

The site's brand accents are `#5eb761` / `#43a047` (green) and `#4284ce`
(blue). The blue and green above are the darker variants the site uses for
text, chosen so that 11–12 px labels clear a 4.5:1 contrast ratio on white.

Filled shapes use the stroke color at `fill-opacity` 0.06–0.18.

## Type

```
sans: 'IBM Plex Sans','Segoe UI',Roboto,Helvetica,Arial,sans-serif
mono: 'IBM Plex Mono',Consolas,'Courier New',monospace
```

Neither webfont is loaded by an `<img>`-embedded SVG, so the fallbacks are what
most readers actually see — keep them in the stack.

Sizes in use: 17 px/600 titles, 13 px subtitles, 11.5 px labels, 12.5 px code,
11 px inline mono.

## Layout

- Outer card: `rect` with `rx="8"`, inset by 0.5 px, `#d0d7de` border.
- Inner panels: `rx="6"`, `#f6f8fa` fill, `#d8dee4` border.
- Canvas width 780–860; no title text inside the figure — the caption in the
  `.. figure::` directive carries it.
- Every figure needs `role="img"`, `aria-label` and a `<title>`.

## Not themed

`phi_theta.svg`, `azimuth_elevation.svg` and `yaw_pitch_roll.svg` keep
red/green/blue axes — that is the standard X/Y/Z convention, not a theme
choice.

## Roles

Where a figure shows both halves of the transceiver, **transmitter is blue and
receiver is green** (`system_model.svg`, `architecture.svg`,
`baseband_cube.svg`). In the ray-tracing figures blue is the radar and its rays,
green is the environment, and coral is the target of interest.

## XML validity

These are served as standalone `.svg` files, so they are parsed as XML, not
HTML — a lenient HTML preview will hide errors that break the real page. `--`
inside a comment is the easy one to trip over. Check with:

```bash
python -c "import glob,xml.dom.minidom as m; [m.parse(f) for f in glob.glob('assets/*.svg')]"
```
