# Path A prototype: soil organic carbon from sensor-type readings

Pipeline: AfSIS CSVs -> afsis_experiment.py -> export/ (model.joblib + results.json) -> soil_api.py -> web/index.html

STATUS: tested end to end on SYNTHETIC AfSIS-shaped data only. Not yet run on real AfSIS files.
web/index.html currently shows embedded DEMO DATA (synthetic) with a banner until you export real results.

## Weekend checklist
Saturday
1. Install: pip install -r requirements.txt  (and the AWS CLI)
2. Look at the bucket layout:  aws s3 ls --no-sign-request s3://afsis/
3. Download only the tables (skip the large spectra files). Adjust paths to the layout you see:
   aws s3 sync s3://afsis/2009-2013/Georeferences  afsis/2009-2013/Georeferences  --no-sign-request
   aws s3 sync s3://afsis/2009-2013/Wet_Chemistry  afsis/2009-2013/Wet_Chemistry  --no-sign-request
4. See what is in the data and send the output back for column fixes:
   python afsis_experiment.py --root afsis/2009-2013 --list-columns
5. Order sensors; line up an oven and scale (oven-dried moisture), pH 4 and 7 buffers, an EC standard.
Sunday
6. Run the real experiment and export (set --target and --features from step 4):
   python afsis_experiment.py --root afsis/2009-2013 --countries Uganda Kenya Tanzania Rwanda --target "<col>" --features "<c1>" "<c2>" --log-target --export export
7. Start the app:  MODEL_DIR=export uvicorn soil_api:app --reload   then open http://127.0.0.1:8000/
8. Record the data card (rows, countries, columns) and the break-even level.

## Notes
- Columns are auto-detected only as a guess; always check what the script prints.
- The interval is the 90th percentile of absolute spatial-CV residuals, not a formal guarantee.
- AfSIS licence: the AWS registry says ODbL (attribution), the dataset README says ODC-By. Read the LICENSE file and credit AfSIS.
