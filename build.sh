#!/bin/bash
# Create SSL directory if it doesn't exist
mkdir -p /etc/ssl/certs

# Copy TiDB CA certificate to the correct location
cp tidb-ca.pem /etc/ssl/certs/tidb.pem