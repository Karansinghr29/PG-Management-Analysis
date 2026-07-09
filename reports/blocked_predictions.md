# Blocked Prediction Tasks — why they cannot be built on real data

These were requested but are **not buildable without inventing data**. Per the
project rule (no mocked mappings, no fake relationships), they are documented and
skipped rather than faked. Each entry states the exact missing join and the one
data change that would unblock it.

| # | Requested task | Why it is blocked (verified) | Unblock condition |
|---|---|---|---|
| 1 | **Tenant exit / notice prediction** | `notices` (25 rows) identifies tenants by `full_name` + `phone` + `apartment_code`/`bed_code`. `invoices` identify tenants by UUID `tenant_id`. There is **no key** linking a UUID to a name/phone/bed, so the label ("this tenant exited") cannot be attached to the billing history that would predict it. | An ID crosswalk mapping `invoices.tenant_id` → tenant name/phone (or → bed). |
| 2 | **Per-bed / per-apartment profitability & occupancy history** | `invoices` carry **no `apartment_code` or `bed_code`** (confirmed: columns absent). Revenue therefore cannot be allocated to a bed or apartment. | Add `apartment_code`/`bed_code` (or a tenant→bed map) to the invoice feed. |
| 3 | **Tenant ↔ maintenance behaviour** (does a tenant who raises many tickets pay late / leave?) | `tickets.tenant_name` is free-text; `invoices.tenant_id` is a UUID. **No reliable key** joins them (name matching would be fabricated). | Same crosswalk as #1, or a `tenant_id` column on tickets. |
| 4 | **Historical bed-level occupancy / vacancy forecasting** | `beds_snapshot` (192) and `beds_catalog` (391) are **current-state snapshots** with no time dimension. There is no per-bed occupancy time series to learn from. | Periodic bed-status snapshots (a dated occupancy log). |
| 5 | **Asset-linked maintenance / depreciation by location** | `assets.apartment_code` is **100% NULL**; `purchase_date`/`purchase_price` are 82% NULL. Assets cannot be tied to an apartment or costed reliably. | Populate `apartment_code` and purchase fields on the asset register. |

## What IS built on real data instead

- **Late-payment prediction** — `is_unpaid` from invoice fields + real per-`tenant_id`
  lagged history (grouping the invoices table on its own key — a real relationship).
- **Revenue / electricity / tenant-count forecasting** — from the real 40-month
  `property_month` series.
- **Tenant segmentation & lifetime-value** — from real per-`tenant_id` billing
  aggregates (tenure, total billed, unpaid ratio). No cross-table join needed.
- **Anomaly detection** — IsolationForest on real electricity and invoice columns.

> Note on occupancy: monthly **active billed-tenant count** (from invoices) is a
> real, valid occupancy *proxy* and is forecast. Physical **bed-level** occupancy
> over time (task #4) remains blocked.
