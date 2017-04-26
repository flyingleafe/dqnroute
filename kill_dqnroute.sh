#!/usr/bin/env bash
ps ax | grep dqnroute | awk '{print $1;}' | xargs kill -9  
