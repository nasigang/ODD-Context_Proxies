# 03. Feature Engineering and ODD-Context Proxies

Implements the construct-controlled feature representations and ego-centric target extractors.

## Feature Taxonomy
1. **Physical Baseline ($P_{\text{clean}}$, 12 features)**:
   - Nearest-clearance SDC--actor dyad kinematics: range, range-rate, longitudinal/lateral relative positions, heading delta, focal speed, and instantaneous focal TTC.
2. **Operational Design Domain Context Proxies ($E_{\text{all}}$, 17 features)**:
   - Static Geometry ($E_{\text{static}}$, 5 features): lane count, road curvature, distance to intersection, speed limit, road type.
   - Dynamic Actor Composition ($E_{\text{comp}}$, 6 features): nearby vehicle count, pedestrian/cyclist density, traffic volume, surrounding mean velocity.
   - Dynamic Interaction ($E_{\text{interact}}$, 6 features): secondary minimum TTC, closing rates of non-focal actors, traffic signal conflict state.
3. **Reference Target**:
   - Ego-centric minimum swept SAT OBB-TTC $\le 3.0$ s across all dynamic actors within 70 m of the SDC, with continuous severity $C_{s,t} = \max(0, 1 - \text{min\_TTC}/3.0)$.
