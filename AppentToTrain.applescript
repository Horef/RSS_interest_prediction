property file_path : null

on run
	set file_path to "/Users/sergiyhoref/Programming/RSS_Interest_Tagger/article_train.txt"
	set file_path to POSIX path of file_path
	
	set interest to 1
	
	tell application "NetNewsWire"
		set app_props to get properties
		(*
		# For debugging purposes
		set app_accounts to get accounts of application "NetNewsWire"
		log "App Accounts"
		log app_accounts
		
		# Notice that only the feeds that are not in folders can be found
		set app_webFeeds to get webFeeds of application "NetNewsWire"
		log "App Web-Feeds"
		log app_webFeeds
		
		set exs_accounts to accounts of app_props
		log exs_accounts
		*)
		
		set current_article to current article of app_props
		set article_title to title of current_article
		log "Title of current article"
		log article_title
		
		# to get the full set of properties	
		set article_properties to get properties of current_article
		#log article_properties
		
		set article_contents to html of current_article
		log "Article Contents"
		log article_contents
		
	end tell
	
	# set processed_title to replace_chars(article_title, ",", "")
	# addTextToFile(processed_title & "," & interest)
	
end run

on addTextToFile(text_app)
	log text_app
	do shell script "echo " & quoted form of text_app & " >> " & quoted form of file_path
	log "Successfully Added"
end addTextToFile

# Function to replace all given chars in a given string
# Source - http://www.macosxautomation.com/applescript/sbrt/sbrt-06.html
# Usage - replace_chars(message_string, "string_to_be_replaced", "replacement_string")
on replace_chars(this_text, search_string, replacement_string)
	set AppleScript's text item delimiters to the search_string
	set the item_list to every text item of this_text
	set AppleScript's text item delimiters to the replacement_string
	set this_text to the item_list as string
	set AppleScript's text item delimiters to ""
	return this_text
end replace_chars