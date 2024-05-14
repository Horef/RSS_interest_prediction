on run
	
	set file_path to "/Users/sergiyhoref/Programming/RSS_Interest_Tagger/predict.py"
	set file_path to POSIX path of file_path
	
	tell application "NetNewsWire"
		set app_props to get properties
		
		set selected_articles to selected articles of app_props
		repeat with current_article in selected_articles
			set article_title to title of current_article
			
			(*
			To find the location of conda executable:
			https://docs.anaconda.com/free/working-with-conda/configurations/python-path/#:~:text=You%20can%20search%20for%20the,in%20the%20active%20conda%20environment.
			*)
			set script_res to do shell script "/Users/sergiyhoref/anaconda3/envs/RSS_interest/bin/python " & quoted form of file_path & " " & quoted form of article_title
			
			set res_val to script_res as number
			
			if res_val = 1 then
				tell application "NetNewsWire" to set the read of current_article to false
			else
				tell application "NetNewsWire" to set the read of current_article to true
			end if
			log "Title of current article"
			log article_title
			log "Set to"
			log script_res
		end repeat
		
	end tell
end run