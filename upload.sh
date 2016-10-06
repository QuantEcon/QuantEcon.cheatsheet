#!/bin/bash

# Sync to S3 Bucket cheatsheet.quantecon.org
aws s3 sync _build/html/ s3://cheatsheets.quantecon.org/ --region=ap-southeast-2  #--delete --dryrun