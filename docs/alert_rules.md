# Alert Rules & Marketing Trigger Logic

## Severity Tiers

| Tier | Score Range | Min Duration | Offer | Channel |
|---|---|---|---|---|
| Mild | 0.20–0.40 | 15 min | 1GB bonus data | Push notification |
| Moderate | 0.40–0.65 | 30 min | 5GB + $3 credit | SMS |
| Severe | 0.65–0.85 | 60 min | Free day pass | SMS |
| Critical | > 0.85 | Any | 1 week free | Personal call |

## Rate Limiting
- Max 1 offer per subscriber per day
- Max 3 offers per subscriber per week
- 24-hour cooldown between alerts for same site

## Zone-Level Alerts
- Triggered when ≥2 sites in same H3 zone are anomalous
- Prevents duplicate offers to subscribers near multiple affected sites
