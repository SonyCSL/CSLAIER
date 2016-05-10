#!/bin/bash
touch deepstation.db
sqlite3 deepstation.db < $(pwd)/scheme/deepstation.sql