---
agent: agent
tools: ['edit', 'search']
---
Ignore all previous context.
If ${selection} is inside a class method and it uses 3 or fewer different instance variables then extract it to a standalone function with the instance variables passed as parameters otherwise extract it to a new method within the class. Name the new function/method according to its functionality and starting with an underscore. Update the original method to call the new function/method with the correct arguments and handle its return value appropriately.
Do not consider any errors/problems that existed before the extraction.
Be very careful with indentation.
