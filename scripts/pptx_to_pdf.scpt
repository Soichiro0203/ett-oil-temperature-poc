on run argv
  set inPath to POSIX file (item 1 of argv)
  set outPath to POSIX file (item 2 of argv)
  tell application "Microsoft PowerPoint"
    with timeout of 600 seconds
      open inPath
      set theDoc to active presentation
      save theDoc in outPath as save as PDF
      close theDoc saving no
    end timeout
  end tell
end run
