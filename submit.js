async function submit_link() {
    var link = document.getElementById("car_link").value;
    if (link !== "") {
        const response = await fetch(`http://127.0.0.1:5000/submit?img_url=${link}`, {
        method: 'POST',
        headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        "enctype": "multipart/form-data"
        }
        });

        response.json().then(data => {
        console.log(data);
        });}

    }