on run(argv)
    tell application "Safari"
        set {first_url, rest_urls} to {item 1 of argv, rest of argv}
        -- set {first_url, rest_urls} to {item 1 of the_urls, rest of the_urls}
        make new document at end of documents with properties {URL:first_url}
        tell window 1
            repeat with the_url in rest_urls
                make new tab at end of tabs with properties {URL:the_url}
            end repeat
        end tell
    end tell
end run