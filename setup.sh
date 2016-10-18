#!/bin/bash
touch cslaier.db
sqlite3 cslaier.db < $(pwd)/scheme/cslaier.sql