"""Prep for Outbrain Click Prediction: join the three needed files into one
impression event table (user_id, campaign_id, clk, day, ts) for replay.

Files needed (competition download, NOT page_views):
  clicks_train.csv    display_id, ad_id, clicked
  events.csv          display_id, uuid, timestamp, platform, geo_location, ...
  promoted_content.csv ad_id, document_id, campaign_id, advertiser_id
Timestamp is ms since 1465876799998 (2016-06-14 04:00 UTC); day = ts // 86400000.
Refuses to overwrite an existing prep dir. Prints schema-validation stats.
"""
import os, sys, json, zipfile
import duckdb

BASE = r"E:\Projects\Submitted\MobileAd_Comverse\research\data\outbrain"
OUT = os.path.join(BASE, "prep")


def find(stem):
    for ext in (".csv", ".csv.zip", ".zip"):
        p = os.path.join(BASE, stem + ext)
        if os.path.exists(p):
            if p.endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    z.extractall(BASE)
                return os.path.join(BASE, stem + ".csv")
            return p
    sys.exit(f"missing {stem}: have {os.listdir(BASE)}")


def main():
    if os.path.exists(OUT):
        sys.exit(f"refusing to overwrite {OUT}")
    clicks, events, promo = find("clicks_train"), find("events"), find("promoted_content")
    os.makedirs(OUT)
    con = duckdb.connect(os.path.join(OUT, "outbrain.duckdb"))
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute(f"CREATE TABLE clk AS SELECT * FROM read_csv_auto('{clicks}')")
    con.execute(f"""CREATE TABLE ev0 AS SELECT
        CAST(display_id AS BIGINT) AS display_id, uuid,
        CAST(timestamp AS BIGINT) AS timestamp,
        NULLIF(platform, '\\N') AS platform,
        NULLIF(geo_location, '\\N') AS geo_location
      FROM read_csv_auto('{events}', all_varchar=true, nullstr='\\N')""")
    con.execute(f"CREATE TABLE promo AS SELECT * FROM read_csv_auto('{promo}')")

    res = {}
    res["rows_clicks"] = con.execute("SELECT count(*) FROM clk").fetchone()[0]
    res["rows_events"] = con.execute("SELECT count(*) FROM ev0").fetchone()[0]
    res["campaigns"] = con.execute("SELECT count(DISTINCT campaign_id) FROM promo").fetchone()[0]
    res["ctr"] = round(con.execute("SELECT avg(clicked) FROM clk").fetchone()[0], 5)

    con.execute("""CREATE TABLE ev AS
      SELECT e.uuid AS user_id, p.campaign_id, c.clicked AS clk, e.timestamp AS ts,
             CAST(e.timestamp / 86400000 AS INT) AS day,
             e.platform, e.geo_location
      FROM clk c JOIN ev0 e USING (display_id) JOIN promo p USING (ad_id)""")
    res["rows_joined"] = con.execute("SELECT count(*) FROM ev").fetchone()[0]
    res["join_rate"] = round(res["rows_joined"] / res["rows_clicks"], 4)
    res["days"] = con.execute("SELECT day, count(*) FROM ev GROUP BY 1 ORDER BY 1").fetchall()
    res["anon_user_share"] = round(con.execute(
        "SELECT avg(CASE WHEN user_id IS NULL THEN 1.0 ELSE 0 END) FROM ev").fetchone()[0], 4)
    cov = con.execute("""
      WITH imp AS (SELECT campaign_id, count(*) c FROM ev GROUP BY 1),
      r AS (SELECT c, sum(c) OVER (ORDER BY c DESC) cum, sum(c) OVER () tot,
            row_number() OVER (ORDER BY c DESC) rk FROM imp)
      SELECT max(CASE WHEN rk=100 THEN cum*1.0/tot END),
             max(CASE WHEN rk=500 THEN cum*1.0/tot END) FROM r""").fetchone()
    res["top100_coverage"], res["top500_coverage"] = round(cov[0] or 0, 3), round(cov[1] or 0, 3)

    with open(os.path.join(OUT, "prep_stats.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
    ok = res["rows_joined"] > 10_000_000 and len(res["days"]) >= 6 and res["campaigns"] > 1000
    print("VALIDATION:", "PASS" if ok else "CHECK")


if __name__ == "__main__":
    main()
