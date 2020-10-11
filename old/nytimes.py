#!/usr/bin/python

from datetime import datetime
import subprocess

def main():
    d = datetime.now()
    years = [d.year - t for t in range(125, 0, -25)]
    URLs = []
    for year in years:
        if (year < 1981):
            date_string = "%d/%02d/%02d" % (year, d.month, d.day)
            URLs.append("http://timesmachine.nytimes.com/timesmachine/%s/issue.html" % date_string)
        else:
            date_string = "%d%02d%02d" % (year, d.month, d.day)
            URLs.append("http://query.nytimes.com/search/sitesearch/#/*/from%sto%s/" % (date_string, date_string))
    cmd = "osascript ~/Desktop/Programming/scripts/opentabs.scpt " + ' '.join(URLs)
    subprocess.call(cmd, shell = True)

if __name__ == "__main__":
    main()