# Dataset Audit — Vista Heights PG Analytics (Step 1)

Column-level inspection of all 8 real datasets, current dashboard usage, unused
columns, and which new analytics are actually supported. Every "not possible"
below is backed by a real null-rate / missing column — nothing is assumed.

## Per-dataset report

### 1. invoices — `Supabase Snippet Untitled query (6).csv`
- **Rows / Cols:** 6,675 × 13
- **Primary columns:** `invoice_id` (PK), `tenant_id`, `billing_month`,
  `rent_amount`, `electricity_amount`, `total_amount`, `is_unpaid`, `prior_unpaid`
- **Current usage:** revenue KPIs, revenue forecast, ML revenue model, late-payment
  model, collection rate, revenue-at-risk, tenant segmentation.
- **Unused columns:** `property_id` (constant), `invoice_month_num` (dup of month),
  `other_charges` (mostly 0).
- **Insights possible:** already heavily used.
- **In dashboard:** ✅ Yes (core).

### 2. electricity — `Supabase Snippet Untitled query (7).csv`
- **Rows / Cols:** 1,376 × 8
- **Primary columns:** `apartment_code`, `billing_month`, `units_consumed`,
  `unit_cost`, `amount`, `apt_avg_units`, `deviation_from_avg`
- **Current usage:** electricity trend, apartment-wise electricity forecast,
  electricity anomaly detection.
- **Unused columns:** `property` (constant).
- **Insights possible:** apartment electricity ranking, consumption hotspots (NEW).
- **In dashboard:** ✅ Yes (partial → extend with ranking).

### 3. beds_snapshot — `Supabase Snippet Untitled query (8).csv`
- **Rows / Cols:** 192 × 9
- **Primary columns:** `apartment_code`, `bed_code`, `bed_type`, `toilet_type`,
  `bed_status`, `bed_lifecycle_status`, `current_rate`, `gender_allowed`
- **Current usage:** occupancy KPI, bed-lifecycle pie.
- **Unused columns:** `toilet_type`, `gender_allowed`, `bed_type` (partly),
  `current_rate` (only in KPI).
- **Insights possible:** vacant-bed availability board, block-wise vacancy, vacancy
  revenue impact via `current_rate` (NEW).
- **In dashboard:** 🟡 Partial → extend (Available Beds module).

### 4. notices — `Supabase Snippet Untitled query (10).csv`
- **Rows / Cols:** 25 × 8
- **Primary columns:** `notice_date`, `estimated_exit_date`, `apartment_code`,
  `bed_code`, `monthly_rental`, `full_name`, `phone`
- **Current usage:** cleaned only (`notice_period_days`) — **not shown in dashboard**.
- **Unused columns:** all of it in the UI.
- **Insights possible:** monthly notice trend, upcoming vacating beds, notice revenue
  impact (`monthly_rental` — real ₹), apartment-wise notice count (NEW).
- **In dashboard:** ❌ No → build Notice & Exit module.
- **Not possible:** *notice reasons* — there is **no reason/remarks column**. Skip.

### 5. beds_catalog — `Supabase Snippet Untitled query (15).csv`
- **Rows / Cols:** 391 × 8 (359 after dedup)
- **Primary columns:** `apartment_code`, `bed_code`, `bed_type`, `monthly_rate`,
  `status` (Occupied/Vacant), `gender_allowed`
- **Current usage:** none directly in the UI.
- **Insights possible:** cross-check vacancy, rate-by-bed-type.
- **In dashboard:** ❌ No (largely redundant with beds_snapshot).

### 6. assets — `Supabase Snippet Untitled query (16).csv`
- **Rows / Cols:** 1,700 × 12
- **Primary columns:** `asset_code` (PK), `asset_type`, `category`, `condition`,
  `status`
- **Current usage:** **completely unused.**
- **Unused columns:** every column.
- **Insights possible (real):** total assets, by category, by asset_type, by status
  (allocated/inventory), by condition (good/new); purchase-by-month + purchase-price
  for the **18% of rows** that have `purchase_date`/`purchase_price` (NEW).
- **In dashboard:** ❌ No → build Asset Management module.
- **Not possible / must skip (with reason):**
  - **Warranty alerts** — `warranty_expiry` is populated on **1 of 1,700 rows
    (0.1%)**. Cannot compute expired / 30-60-90-day buckets. Skip.
  - **Assets by apartment** — `apartment_code` is **100% NULL**. Skip.
  - **Active vs Damaged** — `condition` only takes values `good`/`new` (no "damaged"),
    `status` is `allocated`/`inventory` (no "active/damaged"). Show the real
    distributions instead and note the requested labels don't exist.

### 7. meters — `Supabase Snippet Untitled query (17).csv`
- **Rows / Cols:** 41 × 7 (40 after dedup)
- **Primary columns:** `apartment_code`, `eb_meter_number`, `eb_connection_type`
- **Current usage:** none in the UI.
- **Unused columns:** `eb_card_number` (100% null), `eb_consumer_number` (100% null),
  `eb_sanctioned_load` (constant 0).
- **Insights possible:** minor — connection-type split. Low value; not prioritised.
- **In dashboard:** ❌ No.

### 8. tickets — `Supabase Snippet Untitled query (18).csv`
- **Rows / Cols:** 1,416 × 17
- **Primary columns:** `ticket_number` (PK), `apartment_code`, `issue_type`,
  `priority`, `status`, `created_at`, `resolved_at`, `closed_at`, `sla_deadline`
- **Current usage:** open-ticket KPI, ticket analysis (advanced EDA), anomaly.
- **Unused columns:** `assigned_to` (100% null), `tenant_phone` (2% null),
  `issue_sub_type` (16%), `tenant_approved` (90% — approval rate possible).
- **Insights possible:** full maintenance-performance board — status/priority/issue
  mix, apartment-wise complaints, monthly trend, avg resolution time, SLA breaches
  (NEW consolidated tab).
- **In dashboard:** 🟡 Partial → build Maintenance Performance module.

## Summary of usage state

| State | Datasets |
|---|---|
| **Fully used** | invoices |
| **Partially used** | electricity, beds_snapshot, tickets |
| **Completely unused** | assets, notices, beds_catalog, meters |

**Columns never used anywhere (real, in-scope):** assets → all 12; notices → all 8
(UI); meters → all 7; beds_catalog → all 8; tickets → `assigned_to`,
`issue_sub_type`, `tenant_approved`; beds_snapshot → `toilet_type`, `gender_allowed`.

## New modules that ARE fully supported by real data
1. **Asset Management** (assets) — counts, category/type/status/condition, purchase
   timeline (18% subset, labelled).
2. **Available Beds** (beds_snapshot) — vacant beds, block-wise vacancy, vacancy %,
   vacancy revenue impact via `current_rate`.
3. **Maintenance Performance** (tickets) — status/priority/issue, apartment
   complaints, monthly trend, resolution time, SLA breaches.
4. **Apartment Performance** (electricity + tickets + beds) — electricity, complaints,
   vacancy per apartment + a composite health score.
5. **Notice & Exit** (notices) — monthly trend, upcoming exits, revenue impact,
   apartment-wise count.
6. **Extra insights** — electricity consumption ranking, vacancy hotspots, complaint
   hotspots.

## Requested analytics that CANNOT be built (missing real data)
| Requested | Blocker |
|---|---|
| Warranty expiry alerts (30/60/90) | `warranty_expiry` 0.1% populated |
| Assets by apartment | `apartment_code` 100% null in assets |
| Active vs Damaged assets | no "damaged" value exists (only good/new, allocated/inventory) |
| Floor-wise vacancy | no floor column; only block letter is derivable — shown as **block-wise** |
| Vacancy trend (historical) | beds are a current snapshot, no time history |
| Apartment-wise **revenue / collection** | invoices carry no `apartment_code` (UUID tenant only) |
| Notice reasons | no reason/remarks column in notices |
| Maintenance **cost** distribution | tickets have no cost column; asset `purchase_price` 18% only |
