# Create a new HTTP Client
Add-Type -AssemblyName "System.Net.Http"
$file_name=$args[0]
$Url = "http://127.0.0.1:5000/predict"
$FilePath = "C:\Users\Lucas\Documents\USP\RP2-EI\image_recognition\testing\images\$file_name"

$HttpClient = New-Object System.Net.Http.HttpClient
$MultipartFormDataContent = New-Object System.Net.Http.MultipartFormDataContent

# Create the file content to be sent in the POST request
$FileStream = [System.IO.File]::OpenRead($FilePath)
$FileStreamContent = New-Object System.Net.Http.StreamContent($FileStream)
$FileStreamContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/jpeg")

# Add the file to the form data
$MultipartFormDataContent.Add($FileStreamContent, "file", [System.IO.Path]::GetFileName($FilePath))

# Send the POST request
$Response = $HttpClient.PostAsync($Url, $MultipartFormDataContent).Result
$ResponseContent = $Response.Content.ReadAsStringAsync().Result

# Output the response from the server
$ResponseContent
