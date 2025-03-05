# Glossary

## Non-Wear Time Metrics
**`NonwearTime(days)`**  
- *Description*: Total duration when the device was not worn  
- *Units*: Days  
- *Note*: 0.0 indicates full compliance  

**`NumNonwearEpisodes`**  
- *Description*: Number of periods when device was removed  
- *Units*: Count  
- *Note*: 0 indicates continuous wear  

## Sunlight Exposure Metrics
**`TotalSunlight(mins)`**  
- *Description*: Total sunlight exposure duration
- *Units*: Minutes

**`SunlightDay[Avg/Med/Min/Max](mins)`**  
- *Description*: Daily sunlight statistics
  - Avg: Mean daily duration
  - Med: Median daily duration
  - Min: Minimum daily duration
  - Max: Maximum daily duration

## Movement Metrics
**`ENMO(mg)`**  
- *Description*: Euclidean Norm Minus One - acceleration-based movement intensity  
- *Units*: Milli-gravities (mg)
- *Note*: Higher values indicate more intense movement  

**`TotalMVPA(mins)`**  
- *Description*: Total Moderate-Vigorous Physical Activity duration  
- *Units*: Minutes

**`MVPADay[Avg/Med/Min/Max](mins)`**  
- *Description*: Daily MVPA statistics
  - Avg: Mean daily duration
  - Med: Median daily duration
  - Min: Minimum daily duration
  - Max: Maximum daily duration


## Simultaneous Sunlight + MVPA
**`TotalSunlightMVPA(mins)`**  
- *Description*: Total sunlight exposure + MVPA duration
- *Units*: Minutes
- *Note*: Measures simultaneous sunlight exposure *and* MVPA. Proxy for outdoor physical activity.

**`SunlightMVPADay[Avg/Med/Min/Max](mins)`**  
- *Description*: Daily sunlight exposure + MVPA statistics
  - Avg: Mean daily duration
  - Med: Median daily duration
  - Min: Minimum daily duration
  - Max: Maximum daily duration

## Notes
1. All values are derived using a minute-level resolution.
1. `*Adjusted` values compensate for missing/non-wear data.
1. MVPA = Moderate to Vigorous Physical Activity (>= 100 mg).
