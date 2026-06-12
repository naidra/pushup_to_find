export default function AllSvgs() {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" style={{ display: 'none' }}>
            <defs>
                <filter id="blur-and-scale" y="-50%" x="-50%" width="200%" height="200%">
                    <feGaussianBlur in="SourceGraphic" stdDeviation="20" result="blurred" />
                    <feColorMatrix type="saturate" in="blurred" values="4" />
                    <feComposite in="SourceGraphic" operator="over" />
                </filter>
            </defs>
        </svg>
    );
}
